from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


class ZOSConnectionError(RuntimeError):
    pass


@dataclass
class ZOSSession:
    app: object
    system: object
    ZOSAPI: object

    def close(self) -> None:
        if self.app is not None:
            self.app.CloseApplication()
            self.app = None

    def __enter__(self) -> "ZOSSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def connect(install_dir: Path) -> ZOSSession:
    install_dir = install_dir.resolve()
    helper = install_dir / "ZOSAPI_NetHelper.dll"
    zosapi = install_dir / "ZOSAPI.dll"
    interfaces = install_dir / "ZOSAPI_Interfaces.dll"
    for required in (helper, zosapi, interfaces):
        if not required.is_file():
            raise ZOSConnectionError(f"Missing ZOS-API assembly: {required}")

    try:
        import clr
    except ImportError as exc:
        raise ZOSConnectionError("pythonnet is not installed in this Python environment") from exc

    helper_dir = str(helper.parent)
    if helper_dir not in sys.path:
        sys.path.append(helper_dir)
    clr.AddReference(str(helper))
    import ZOSAPI_NetHelper

    if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(str(install_dir)):
        raise ZOSConnectionError(f"ZOSAPI initializer rejected install directory: {install_dir}")
    zemax_dir = Path(str(ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()))
    clr.AddReference(str(zemax_dir / "ZOSAPI_Interfaces.dll"))
    clr.AddReference(str(zemax_dir / "ZOSAPI.dll"))
    import ZOSAPI

    connection = ZOSAPI.ZOSAPI_Connection()
    app = connection.CreateNewApplication()
    if app is None:
        raise ZOSConnectionError("CreateNewApplication returned no application")
    if not bool(app.IsValidLicenseForAPI):
        status = str(app.LicenseStatus)
        app.CloseApplication()
        raise ZOSConnectionError(f"OpticStudio license is not valid for ZOS-API (status: {status})")
    return ZOSSession(app=app, system=app.PrimarySystem, ZOSAPI=ZOSAPI)


def installation_report(install_dir: Path) -> dict[str, object]:
    root = install_dir.resolve()
    libraries = root
    files = ["ZOSAPI_NetHelper.dll", "ZOSAPI.dll", "ZOSAPI_Interfaces.dll"]
    return {
        "install_dir": str(root),
        "opticstudio_exists": (root / "OpticStudio.exe").is_file(),
        "assemblies": {name: (libraries / name).is_file() for name in files},
    }
