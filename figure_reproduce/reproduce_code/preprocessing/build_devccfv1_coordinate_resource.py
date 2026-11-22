from __future__ import annotations
import argparse
import gzip
import json
import logging
import os
import re
import struct
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
SCRIPT_PATH = Path(__file__).resolve()
DATASET_KEY = 'DevCCFv1_figshare_26377171'
RAW_ROOT = Path(__file__).resolve().parents[2] / 'Datasets' / 'Raw'
PROCESSED_ROOT = Path(__file__).resolve().parents[2] / 'Datasets' / 'Processed'
REGION_SCHEMA = [(0, 'Background', '#000000', False, False, False, 'Atlas background', 'atlas_only'), (1, 'DorsalVZ', '#cc4c02', True, True, True, 'Dorsal cortical ventricular/periventricular zone', ''), (2, 'IZ', '#756bb1', True, True, True, 'Cortical intermediate zone', ''), (3, 'CP', '#2b8cbe', True, True, True, 'Cortical plate / superficial cortical stratum', ''), (4, 'VentralVZ', '#fd8d3c', True, True, True, 'Ventral ganglionic ventricular zone', ''), (5, 'GE', '#31a354', True, True, True, 'Non-VZ ganglionic eminence', ''), (6, 'SeptalVZ', '#e6550d', True, True, True, 'Septal ventricular zone', ''), (7, 'Septum', '#636363', True, True, True, 'Non-VZ septal domain', ''), (8, 'BT', '#de2d26', True, True, True, 'Basal/lateral telencephalon residual', ''), (9, 'Other', '#d9d9d9', True, True, False, 'Non-target or unresolved annotated tissue', '')]
CP_ACRONYMS = {'NeoCxs', 'MesoCxs', 'HCCxs'}
IZ_ACRONYMS = {'NeoCxi', 'MesoCxi', 'HCCxi'}
DORSAL_VZ_ACRONYMS = {'NeoCxp', 'NeoCxv', 'MesoCxp', 'MesoCxv', 'HCCxp', 'HCCxv'}
VENTRAL_VZ_ACRONYMS = {'Dgv', 'Palv', 'Strv', 'AStrv', 'APalv', 'ADgv'}
SEPTAL_VZ_ACRONYMS = {'DgSev', 'PalSev', 'StrSev', 'SeDgv', 'SePalv', 'SeStrv'}
SEPTUM_PREFIXES = ('DgSe', 'PalSe', 'StrSe', 'SeDg', 'SePal', 'SeStr')
GE_ACRONYMS = {'Dg', 'Dgp', 'Dgi', 'Dgs', 'Pal', 'Palp', 'Pali', 'Pals', 'Str', 'Strp', 'Stri', 'Strs', 'AStri', 'AStrs'}
BT_EXACT_ACRONYMS = {'POA', 'SPall', 'APall'}
BT_PREFIXES = ('APal', 'ADg', 'OlfCx')
NON_TELENCEPHALON_PREFIXES = ('THy', 'PHy', 'p1', 'p2', 'p3', 'm1', 'm2', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'Sp')
DTYPE_BY_NIFTI = {2: np.dtype('u1'), 4: np.dtype('i2'), 8: np.dtype('i4'), 16: np.dtype('f4'), 64: np.dtype('f8'), 256: np.dtype('i1'), 512: np.dtype('u2'), 768: np.dtype('u4')}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build DevCCFv1 coordinate-system resources.')
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--processed-root', type=Path, default=PROCESSED_ROOT)
    parser.add_argument('--dataset-key', default=DATASET_KEY)
    parser.add_argument('--ages', nargs='+', default=['E15.5', 'E18.5'])
    parser.add_argument('--build-all-ages', action='store_true')
    parser.add_argument('--build-gse269617-crosswalk', action='store_true')
    parser.add_argument('--write-voxel-tables', action='store_true')
    parser.add_argument('--write-qc-figures', action='store_true')
    parser.add_argument('--max-voxels-per-age', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'build_{DATASET_KEY}_{stamp}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return log_path

def paths(processed_root: Path, dataset_key: str) -> dict[str, Path]:
    root = processed_root / dataset_key / 'coordinate_system'
    out = {'root': root, 'extracted': root / 'extracted', 'tables': root / 'tables', 'volumes': root / 'volumes', 'voxels': root / 'voxel_tables', 'merged': root / 'GSE269617_region_merged', 'figures': root / 'figures', 'manifest': root / 'manifest', 'logs': root / 'logs'}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out

def parse_itksnap_labels(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile('^\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s+([0-9.]+)\\s+(\\d+)\\s+(\\d+)\\s+"(.*)"\\s*$')
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = pattern.match(line)
        if not match:
            continue
        struct_id, red, green, blue, alpha, visible, mesh_visible, label_text = match.groups()
        if ': ' in label_text:
            acronym, name = label_text.split(': ', 1)
        else:
            acronym, name = (label_text, label_text)
        rows.append({'struct_id': int(struct_id), 'red': int(red), 'green': int(green), 'blue': int(blue), 'alpha': float(alpha), 'visible': int(visible), 'mesh_visible': int(mesh_visible), 'label_text': label_text, 'acronym': acronym.strip(), 'name': name.strip(), 'source_file': str(path)})
    if not rows:
        raise ValueError(f'No ITK-SNAP labels parsed from {path}')
    return pd.DataFrame(rows)

def parse_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception as exc:
        logging.info('Falling back to minimal xlsx parser for %s: %s', path, exc)
    with zipfile.ZipFile(path) as zf:
        shared = read_shared_strings(zf)
        sheet_name = first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet_name))
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows: list[list[Any]] = []
    for row in root.findall('.//a:sheetData/a:row', ns):
        values: list[Any] = []
        last_col = -1
        for cell in row.findall('a:c', ns):
            ref = cell.attrib.get('r', '')
            col = column_index(ref)
            while last_col + 1 < col:
                values.append('')
                last_col += 1
            values.append(read_xlsx_cell(cell, shared, ns))
            last_col = col
        rows.append(values)
    if not rows:
        return pd.DataFrame()
    width = max((len(row) for row in rows))
    rows = [row + [''] * (width - len(row)) for row in rows]
    header = [str(x).strip() if str(x).strip() else f'column_{i}' for i, x in enumerate(rows[0])]
    return pd.DataFrame(rows[1:], columns=header)

def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if 'xl/sharedStrings.xml' not in zf.namelist():
        return []
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    strings = []
    for si in root.findall('a:si', ns):
        parts = [node.text or '' for node in si.findall('.//a:t', ns)]
        strings.append(''.join(parts))
    return strings

def first_sheet_path(zf: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(zf.read('xl/workbook.xml'))
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    rel_by_id = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels if rel.tag.endswith('Relationship') and 'Id' in rel.attrib}
    for sheet in workbook.iter():
        if sheet.tag.endswith('sheet'):
            rel_id = sheet.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target = rel_by_id.get(rel_id, 'worksheets/sheet1.xml')
            return 'xl/' + target.lstrip('/')
    return 'xl/worksheets/sheet1.xml'

def column_index(ref: str) -> int:
    letters = ''.join((ch for ch in ref if ch.isalpha()))
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord('A') + 1)
    return max(0, value - 1)

def read_xlsx_cell(cell: ET.Element, shared: list[str], ns: dict[str, str]) -> Any:
    cell_type = cell.attrib.get('t')
    value = cell.find('a:v', ns)
    if cell_type == 'inlineStr':
        parts = [node.text or '' for node in cell.findall('.//a:t', ns)]
        return ''.join(parts)
    if value is None or value.text is None:
        return ''
    text = value.text
    if cell_type == 's':
        idx = int(float(text))
        return shared[idx] if idx < len(shared) else text
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text

def normalize_ontology(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    out = df.copy()
    lowered = {str(c).strip().lower(): c for c in out.columns}

    def find_col(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        for key, col in lowered.items():
            if any((candidate.lower() in key for candidate in candidates)):
                return col
        return None
    id_col = find_col('struct_id', 'structure id', 'id', 'IDX')
    acronym_col = find_col('acronym', 'abbreviation')
    name_col = find_col('name', 'full name', 'label')
    parent_col = find_col('parent_struct_id', 'parent id', 'parent')
    if 'struct_id' not in out:
        out['struct_id'] = pd.to_numeric(out[id_col], errors='coerce') if id_col else np.nan
    if 'acronym' not in out:
        out['acronym'] = out[acronym_col].astype(str) if acronym_col else ''
    if 'name' not in out:
        out['name'] = out[name_col].astype(str) if name_col else ''
    if 'parent_struct_id' not in out:
        out['parent_struct_id'] = pd.to_numeric(out[parent_col], errors='coerce') if parent_col else np.nan
    out['source_file'] = str(source_file)
    return out

def choose_annotation_member(zf: zipfile.ZipFile, age: str) -> str:
    prefix = age + '/'
    candidates = [name for name in zf.namelist() if name.startswith(prefix) and 'DevCCF_Annotations' in name and name.endswith('.nii.gz')]
    if not candidates:
        raise FileNotFoundError(f'No DevCCF annotation NIfTI found in {zf.filename} for {age}')
    preferred = [name for name in candidates if '20um' in name]
    return sorted(preferred or candidates)[0]

def read_nifti_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        compressed = zf.read(member)
    raw = gzip.decompress(compressed)
    return read_nifti_bytes(raw, zip_path, member)

def read_nifti_bytes(raw: bytes, zip_path: Path, member: str) -> dict[str, Any]:
    endian = '<'
    if struct.unpack('<i', raw[:4])[0] != 348:
        if struct.unpack('>i', raw[:4])[0] == 348:
            endian = '>'
        else:
            raise ValueError(f'{member} is not a NIfTI-1 file')
    dim = struct.unpack(endian + '8h', raw[40:56])
    ndim = int(dim[0])
    shape = tuple((int(x) for x in dim[1:1 + ndim]))
    datatype = struct.unpack(endian + 'h', raw[70:72])[0]
    bitpix = struct.unpack(endian + 'h', raw[72:74])[0]
    pixdim = struct.unpack(endian + '8f', raw[76:108])
    vox_offset = int(round(struct.unpack(endian + 'f', raw[108:112])[0]))
    scl_slope = struct.unpack(endian + 'f', raw[112:116])[0]
    scl_inter = struct.unpack(endian + 'f', raw[116:120])[0]
    sform_code = struct.unpack(endian + 'h', raw[254:256])[0]
    if datatype not in DTYPE_BY_NIFTI:
        raise ValueError(f'Unsupported NIfTI datatype {datatype} in {member}')
    dtype = DTYPE_BY_NIFTI[datatype].newbyteorder(endian)
    n_values = int(np.prod(shape))
    start = vox_offset
    stop = start + n_values * dtype.itemsize
    data = np.frombuffer(raw[start:stop], dtype=dtype, count=n_values).reshape(shape, order='F')
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        data = data.astype(np.float32) * (scl_slope if scl_slope != 0 else 1.0) + scl_inter
    affine = np.eye(4, dtype=float)
    if sform_code > 0:
        affine[0, :] = struct.unpack(endian + '4f', raw[280:296])
        affine[1, :] = struct.unpack(endian + '4f', raw[296:312])
        affine[2, :] = struct.unpack(endian + '4f', raw[312:328])
    else:
        affine[0, 0] = pixdim[1] or 1.0
        affine[1, 1] = pixdim[2] or 1.0
        affine[2, 2] = pixdim[3] or 1.0
    return {'raw': raw, 'endian': endian, 'data': np.asarray(np.rint(data), dtype=np.int64), 'shape': shape, 'datatype': int(datatype), 'bitpix': int(bitpix), 'pixdim': pixdim, 'vox_offset': vox_offset, 'affine': affine, 'source_zip': str(zip_path), 'source_member': member}

def write_uint8_nifti_like(template: dict[str, Any], data: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = template['raw']
    endian = template['endian']
    vox_offset = template['vox_offset']
    header = bytearray(raw[:vox_offset])
    header[70:72] = struct.pack(endian + 'h', 2)
    header[72:74] = struct.pack(endian + 'h', 8)
    header[112:116] = struct.pack(endian + 'f', 1.0)
    header[116:120] = struct.pack(endian + 'f', 0.0)
    header[124:128] = struct.pack(endian + 'f', float(np.max(data)))
    header[128:132] = struct.pack(endian + 'f', float(np.min(data)))
    payload = np.asarray(data, dtype=np.uint8).tobytes(order='F')
    with gzip.open(output_path, 'wb') as fh:
        fh.write(header)
        fh.write(payload)

def write_region_schema(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(REGION_SCHEMA, columns=['region_id', 'region_label', 'hex_color', 'include_in_st_reference', 'include_in_devccf_target', 'include_in_final_generation', 'description', 'notes'])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)
    return df

def infer_broad_region(acronym: str, name: str) -> tuple[int, str, str]:
    ac = str(acronym)
    text = f'{acronym} {name}'.lower()
    if ac == '0' or 'clear label' in text:
        return (0, 'Background', 'zero_or_clear_label')
    if ac in CP_ACRONYMS:
        return (3, 'CP', 'exact_cortical_superficial_stratum')
    if ac in IZ_ACRONYMS:
        return (2, 'IZ', 'exact_cortical_intermediate_stratum')
    if ac in DORSAL_VZ_ACRONYMS:
        return (1, 'DorsalVZ', 'exact_cortical_periventricular_or_vz')
    if ac in VENTRAL_VZ_ACRONYMS:
        return (4, 'VentralVZ', 'exact_ganglionic_vz')
    if ac in SEPTAL_VZ_ACRONYMS:
        return (6, 'SeptalVZ', 'exact_septal_vz')
    if ac.startswith(SEPTUM_PREFIXES):
        return (7, 'Septum', 'exact_non_vz_septal_domain')
    if ac in GE_ACRONYMS:
        return (5, 'GE', 'exact_non_vz_ganglionic_domain')
    if ac in BT_EXACT_ACRONYMS or ac.startswith(BT_PREFIXES):
        return (8, 'BT', 'exact_basal_lateral_telencephalon')
    if ac.startswith(NON_TELENCEPHALON_PREFIXES):
        return (9, 'Other', 'excluded_non_telencephalic_region')
    return (9, 'Other', 'qc_unresolved_or_non_target_region')

def build_crosswalk(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in labels.itertuples(index=False):
        region_id, region_label, rule = infer_broad_region(str(row.acronym), str(row.name))
        rows.append({'struct_id': int(row.struct_id), 'struct_acronym': str(row.acronym), 'struct_name': str(row.name), 'age': 'all', 'ontology_path': '', 'broad_region_id': int(region_id), 'broad_region_label': region_label, 'mapping_rule': rule, 'mapping_source': 'auto_keyword_v1', 'review_status': 'auto_review_required' if region_label == 'Other' and int(row.struct_id) != 0 else 'auto_mapped', 'notes': ''})
    return pd.DataFrame(rows)

def map_volume(data: np.ndarray, crosswalk: pd.DataFrame) -> np.ndarray:
    max_id = int(max(np.max(data), crosswalk['struct_id'].max()))
    lut = np.full(max_id + 1, 9, dtype=np.uint8)
    lut[0] = 0
    for row in crosswalk.itertuples(index=False):
        sid = int(row.struct_id)
        if sid <= max_id:
            lut[sid] = int(row.broad_region_id)
    clipped = np.where(data <= max_id, data, 0)
    return lut[clipped]

def voxel_table(age: str, template: dict[str, Any], broad: np.ndarray, label_lookup: pd.DataFrame, max_voxels: int, seed: int) -> pd.DataFrame:
    indices = np.argwhere(broad > 0)
    if indices.shape[0] > max_voxels:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(indices.shape[0], size=max_voxels, replace=False))
        indices = indices[pick]
    ones = np.ones((indices.shape[0], 1), dtype=float)
    xyz = np.c_[indices.astype(float), ones] @ np.asarray(template['affine'], dtype=float).T
    ids = template['data'][indices[:, 0], indices[:, 1], indices[:, 2]]
    labels = label_lookup.reindex(ids).reset_index(drop=True)
    return pd.DataFrame({'age': age, 'source_volume': template['source_member'], 'voxel_i': indices[:, 0], 'voxel_j': indices[:, 1], 'voxel_k': indices[:, 2], 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 'struct_id': ids.astype(int), 'struct_acronym': labels['acronym'].fillna('').to_numpy(), 'struct_name': labels['name'].fillna('').to_numpy(), 'broad_region_id': broad[indices[:, 0], indices[:, 1], indices[:, 2]].astype(int), 'broad_region_label': [REGION_SCHEMA[int(x)][1] for x in broad[indices[:, 0], indices[:, 1], indices[:, 2]]]})

def write_qc_figure(age: str, broad: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning('Skipping QC figure for %s: %s', age, exc)
        return
    slices = [broad.shape[axis] // 2 for axis in range(3)]
    panels = [broad[slices[0], :, :], broad[:, slices[1], :], broad[:, :, slices[2]]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax, panel, title in zip(axes, panels, ['i-mid', 'j-mid', 'k-mid']):
        ax.imshow(np.rot90(panel), interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
        ax.set_title(f'{age} {title}')
        ax.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def environment_summary() -> dict[str, str]:
    return {'python': sys.version.split()[0], 'executable': sys.executable, 'platform': sys.platform, 'conda_prefix': os.environ.get('CONDA_PREFIX', '')}

def main() -> None:
    args = parse_args()
    raw_dataset = args.raw_root / args.dataset_key
    if args.build_all_ages:
        args.ages = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P04', 'P14', 'P56']
    out = paths(args.processed_root, args.dataset_key)
    setup_logging(out['logs'])
    logging.info('Building DevCCF resource for ages=%s', args.ages)
    label_path = raw_dataset / 'components' / 'ITK-SNAP Label Definitions' / 'DevCCFv1_ITK-SNAP_label.txt'
    labels = parse_itksnap_labels(label_path)
    schema = write_region_schema(out['merged'] / 'gse269617_region_schema.tsv')
    crosswalk = build_crosswalk(labels)
    if args.dry_run:
        inventory = []
        for age in args.ages:
            zpath = raw_dataset / 'components' / f'{age}.zip'
            with zipfile.ZipFile(zpath) as zf:
                inventory.append({'age': age, 'zip': str(zpath), 'annotation_member': choose_annotation_member(zf, age)})
        print(json.dumps({'inventory': inventory, 'schema': schema.to_dict(orient='records')}, indent=2))
        return
    labels.to_csv(out['tables'] / 'devccf_itksnap_labels.tsv', sep='\t', index=False)
    crosswalk.to_csv(out['merged'] / 'devccf_struct_to_gse_region.tsv', sep='\t', index=False)
    for xlsx_name, out_name in [('DevCCFv1_OntologyStructure.xlsx', 'devccf_ontology.tsv'), ('CCFv3_OntologyStructure_u16.xlsx', 'ccfv3_ontology_u16.tsv')]:
        xlsx_path = raw_dataset / 'components' / xlsx_name
        if xlsx_path.exists():
            normalize_ontology(parse_xlsx_first_sheet(xlsx_path), xlsx_path).to_csv(out['tables'] / out_name, sep='\t', index=False)
    label_lookup = labels.set_index('struct_id')
    inventory_rows: list[dict[str, Any]] = []
    for age in args.ages:
        zip_path = raw_dataset / 'components' / f'{age}.zip'
        with zipfile.ZipFile(zip_path) as zf:
            member = choose_annotation_member(zf, age)
        logging.info('%s: reading %s', age, member)
        template = read_nifti_from_zip(zip_path, member)
        broad = map_volume(template['data'], crosswalk)
        broad_path = out['merged'] / f'{age}_broad_region_annotations.nii.gz'
        write_uint8_nifti_like(template, broad, broad_path)
        counts = pd.Series(broad.ravel()).value_counts().sort_index()
        counts_frame = pd.DataFrame({'region_id': counts.index.astype(int), 'region_label': [REGION_SCHEMA[int(i)][1] for i in counts.index], 'voxel_count': counts.to_numpy(dtype=int)})
        counts_frame.to_csv(out['merged'] / f'{age}_broad_region_voxel_counts.tsv', sep='\t', index=False)
        if args.write_voxel_tables:
            vt = voxel_table(age, template, broad, label_lookup, int(args.max_voxels_per_age), int(args.seed))
            vt.to_csv(out['merged'] / f'{age}_nonbackground_voxels.tsv.gz', sep='\t', index=False, compression='gzip')
        if args.write_qc_figures:
            write_qc_figure(age, broad, out['figures'] / f'{age}_broad_region_slices.png')
        inventory_rows.append({'age': age, 'source_zip': str(zip_path), 'source_member': member, 'output_broad_region_volume': str(broad_path), 'shape': 'x'.join((str(x) for x in template['shape'])), 'datatype': template['datatype'], 'bitpix': template['bitpix'], 'pixdim': json.dumps([float(x) for x in template['pixdim']]), 'affine': json.dumps(np.asarray(template['affine']).tolist()), 'nonbackground_voxels': int(np.count_nonzero(broad)), 'regions_present': ','.join((str(int(x)) for x in sorted(np.unique(broad))))})
    pd.DataFrame(inventory_rows).to_csv(out['manifest'] / 'devccf_volume_inventory.tsv', sep='\t', index=False)
    manifest = {'dataset_key': args.dataset_key, 'created_at': now_iso(), 'script': str(SCRIPT_PATH), 'environment': environment_summary(), 'ages': args.ages, 'raw_root': str(raw_dataset), 'processed_root': str(out['root']), 'schema_path': str(out['merged'] / 'gse269617_region_schema.tsv')}
    (out['manifest'] / 'devccf_processing_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logging.info('Done. Wrote %s', out['root'])
if __name__ == '__main__':
    main()
