# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

triton_backends_datas = collect_data_files('triton', subdir='backends', include_py_files=True)


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[("model_milestones", "model_milestones"), ("title.txt", "."), ("help.txt", ".")] + triton_backends_datas,
    hiddenimports=[
        'ffmpeg.nodes',
        'ffmpeg.filters',
        'ffmpeg.streams',
        'ffmpeg.bin',
        'ffmpeg.dag',
        # Add other ffmpeg submodules if needed
        ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=True,
    name='boredealis',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='boredealis',
# )
