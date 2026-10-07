"""Station metadata catalog and region taxonomy for Colombia."""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd


class NaturalRegion(str, Enum):
    """Natural regions of Colombia evaluated in TROPICOR."""

    ANDINA = "ANDINA"
    CARIBE = "CARIBE"
    PACIFICO = "PACIFICO"
    ORINOQUIA = "ORINOQUIA"
    AMAZONIA = "AMAZONIA"
    INSULAR = "INSULAR"


@dataclass(frozen=True)
class StationMetadata:
    """Immutable station metadata record."""

    code: str
    name: str
    latitude: float
    longitude: float
    elevation: float  # meters above sea level (masl)
    region: NaturalRegion
    department: str
    municipality: Optional[str] = None
    variable: str = "PRECIPITACION"

    def __post_init__(self) -> None:
        """Validate spatial coordinates upon initialization."""
        if not (-4.5 <= self.latitude <= 13.5):
            msg = f"Latitude {self.latitude} outside Colombia bounds [-4.5, 13.5]."
            raise ValueError(msg)
        if not (-82.0 <= self.longitude <= -66.5):
            msg = f"Longitude {self.longitude} outside Colombia bounds [-82.0, -66.5]."
            raise ValueError(msg)


# Canonical benchmark station catalog (40 stations evaluated across 6 natural regions)
# Canonical benchmark station catalog (48 stations evaluated across 6 natural regions)
BENCHMARK_STATIONS: List[StationMetadata] = [
    StationMetadata(
        code="52055210",
        name="BOTANA - AUT",
        latitude=1.16,
        longitude=-77.278806,
        elevation=2790.0,
        region=NaturalRegion.ANDINA,
        department="Nariño",
        municipality="Pasto",
    ),
    StationMetadata(
        code="21135050",
        name="JULIA LA",
        latitude=3.098778,
        longitude=-75.530028,
        elevation=1737.0,
        region=NaturalRegion.ANDINA,
        department="Huila",
        municipality="Neiva",
    ),
    StationMetadata(
        code="21205710",
        name="JARDIN BOTANICO - AUT",
        latitude=4.669333,
        longitude=-74.102667,
        elevation=2553.0,
        region=NaturalRegion.ANDINA,
        department="Cundinamarca",
        municipality="Bogotá",
    ),
    StationMetadata(
        code="21205420",
        name="TIBAITATA",
        latitude=4.691417,
        longitude=-74.209,
        elevation=2540.0,
        region=NaturalRegion.ANDINA,
        department="Cundinamarca",
        municipality="Mosquera",
    ),
    StationMetadata(
        code="26215020",
        name="CANAFISANTO",
        latitude=6.416667,
        longitude=-75.816667,
        elevation=480.0,
        region=NaturalRegion.ANDINA,
        department="Antioquia",
        municipality="Santa Fe De Antioquia",
    ),
    StationMetadata(
        code="23095010",
        name="AEROPUERTO PUERTO BERRIO",
        latitude=6.465,
        longitude=-74.412222,
        elevation=110.0,
        region=NaturalRegion.ANDINA,
        department="Antioquia",
        municipality="Puerto Berrío",
    ),
    StationMetadata(
        code="23155040",
        name="CENTRO EL",
        latitude=6.859556,
        longitude=-73.765083,
        elevation=104.0,
        region=NaturalRegion.ANDINA,
        department="Santander",
        municipality="Barrancabermeja",
    ),
    StationMetadata(
        code="23195040",
        name="UNIVERSIDAD INDUSTRIAL SANTANDER",
        latitude=7.144722,
        longitude=-73.122222,
        elevation=898.0,
        region=NaturalRegion.ANDINA,
        department="Santander",
        municipality="Bucaramanga",
    ),
    StationMetadata(
        code="27045020",
        name="CASERI",
        latitude=7.811722,
        longitude=-74.935972,
        elevation=59.0,
        region=NaturalRegion.ANDINA,
        department="Cauca",
        municipality="Cucasia",
    ),
    StationMetadata(
        code="23215030",
        name="AGUAS CLARAS",
        latitude=8.228889,
        longitude=-73.602778,
        elevation=134.0,
        region=NaturalRegion.ANDINA,
        department="Cesar",
        municipality="Aguachica",
    ),
    StationMetadata(
        code="16015040",
        name="SANTA ISABEL",
        latitude=8.233333,
        longitude=-72.433333,
        elevation=140.0,
        region=NaturalRegion.ANDINA,
        department="Norte de Santander",
        municipality="Cúcuta",
    ),
    StationMetadata(
        code="48015050",
        name="AEROPUERTO VASQUEZ COBO",
        latitude=-4.193861,
        longitude=-69.940917,
        elevation=79.0,
        region=NaturalRegion.AMAZONIA,
        department="Amazonas",
        municipality="Leticia",
    ),
    StationMetadata(
        code="47075010",
        name="LA CHORRERA",
        latitude=-1.444611,
        longitude=-72.789583,
        elevation=110.0,
        region=NaturalRegion.AMAZONIA,
        department="Amazonas",
        municipality="La Chorrera",
    ),
    StationMetadata(
        code="47045010",
        name="PUERTO LEGUIZAMO",
        latitude=-0.180611,
        longitude=-74.776278,
        elevation=191.0,
        region=NaturalRegion.AMAZONIA,
        department="Putumayo",
        municipality="Puerto Leguizamo",
    ),
    StationMetadata(
        code="42075010",
        name="MITU",
        latitude=1.26,
        longitude=-70.24,
        elevation=170.0,
        region=NaturalRegion.AMAZONIA,
        department="Vaupés",
        municipality="Mitú",
    ),
    StationMetadata(
        code="44035030",
        name="MACAGUAL",
        latitude=1.5,
        longitude=-75.66,
        elevation=238.0,
        region=NaturalRegion.AMAZONIA,
        department="Caquetá",
        municipality="Florencia",
    ),
    StationMetadata(
        code="46015030",
        name="SAN VICENTE DEL CAGUAN - AUT",
        latitude=2.063,
        longitude=-74.762694,
        elevation=279.0,
        region=NaturalRegion.AMAZONIA,
        department="Caquetá",
        municipality="San Vicente Del Caguan",
    ),
    StationMetadata(
        code="32155010",
        name="MAPIRIPANA",
        latitude=2.8,
        longitude=-70.533333,
        elevation=151.0,
        region=NaturalRegion.AMAZONIA,
        department="Guaviare",
        municipality="San Jose Del Guaviare",
    ),
    StationMetadata(
        code="31095030",
        name="PUERTO INIRIDA - AUT",
        latitude=3.874417,
        longitude=-67.919056,
        elevation=96.0,
        region=NaturalRegion.AMAZONIA,
        department="Guainía",
        municipality="Inírida",
    ),
    StationMetadata(
        code="51035010",
        name="AEROPUERTO LA FLORIDA",
        latitude=1.8,
        longitude=-78.766667,
        elevation=4.0,
        region=NaturalRegion.PACIFICO,
        department="Valle del Cauca",
        municipality="San Andres De Tumaco",
    ),
    StationMetadata(
        code="54075020",
        name="BAJO CALIMA",
        latitude=3.953556,
        longitude=-76.990444,
        elevation=53.0,
        region=NaturalRegion.PACIFICO,
        department="Valle del Cauca",
        municipality="Buenaventura",
    ),
    StationMetadata(
        code="54085010",
        name="NOANAMA",
        latitude=4.68819,
        longitude=-76.93425,
        elevation=21.0,
        region=NaturalRegion.PACIFICO,
        department="Chocó",
        municipality="Medio San Juan",
    ),
    StationMetadata(
        code="54025010",
        name="SAN JOSE PALMAR",
        latitude=4.898083,
        longitude=-76.676667,
        elevation=43.0,
        region=NaturalRegion.PACIFICO,
        department="Chocó",
        municipality="Nóvita",
    ),
    StationMetadata(
        code="11045010",
        name="AEROPUERTO EL CARANO",
        latitude=5.690556,
        longitude=-76.643778,
        elevation=53.0,
        region=NaturalRegion.PACIFICO,
        department="Chocó",
        municipality="Quibdó",
    ),
    StationMetadata(
        code="56015010",
        name="PANAMERICANA",
        latitude=6.223333,
        longitude=-77.404444,
        elevation=12.0,
        region=NaturalRegion.PACIFICO,
        department="Chocó",
        municipality="Bahia Solano",
    ),
    StationMetadata(
        code="11125010",
        name="TERESITA LA",
        latitude=7.0,
        longitude=-77.5,
        elevation=25.0,
        region=NaturalRegion.PACIFICO,
        department="Chocó",
        municipality="Riosucio",
    ),
    StationMetadata(
        code="12025040",
        name="TURBO - AUT",
        latitude=8.091806,
        longitude=-76.716333,
        elevation=10.0,
        region=NaturalRegion.CARIBE,
        department="Antioquia",
        municipality="Turbo",
    ),
    StationMetadata(
        code="12025030",
        name="MELLITO EL",
        latitude=8.542667,
        longitude=-76.673361,
        elevation=52.0,
        region=NaturalRegion.CARIBE,
        department="Chocó",
        municipality="Necoclí",
    ),
    StationMetadata(
        code="13085050",
        name="LORICA ITA - AUT",
        latitude=9.252722,
        longitude=-75.844417,
        elevation=24.0,
        region=NaturalRegion.CARIBE,
        department="Córdoba",
        municipality="Lorica",
    ),
    StationMetadata(
        code="25025002",
        name="LOS ALAMOS - AUT",
        latitude=9.304056,
        longitude=-74.273639,
        elevation=34.0,
        region=NaturalRegion.CARIBE,
        department="Magdalena",
        municipality="San Sebastian De Buenavista",
    ),
    StationMetadata(
        code="28035040",
        name="GUAYMARAL",
        latitude=9.904917,
        longitude=-73.647528,
        elevation=56.0,
        region=NaturalRegion.CARIBE,
        department="Cesar",
        municipality="Valledupar",
    ),
    StationMetadata(
        code="14015030",
        name="ESCUELA NAVAL CIOH",
        latitude=10.447222,
        longitude=-75.516111,
        elevation=6.0,
        region=NaturalRegion.CARIBE,
        department="Bolívar",
        municipality="Cartagena De Indias",
    ),
    StationMetadata(
        code="29045120",
        name="FLORES LAS",
        latitude=11.04,
        longitude=-74.820833,
        elevation=5.0,
        region=NaturalRegion.CARIBE,
        department="Atlántico",
        municipality="Barranquilla",
    ),
    StationMetadata(
        code="15015060",
        name="SAN LORENZO - AUT",
        latitude=11.111083,
        longitude=-74.054694,
        elevation=2254.0,
        region=NaturalRegion.CARIBE,
        department="Magdalena",
        municipality="Santa Marta",
    ),
    StationMetadata(
        code="15075030",
        name="MANAURE",
        latitude=11.781056,
        longitude=-72.4635,
        elevation=0.0,
        region=NaturalRegion.CARIBE,
        department="La Guajira",
        municipality="Manaure",
    ),
    StationMetadata(
        code="32075060",
        name="COOPERATIVA LA",
        latitude=3.366667,
        longitude=-73.7,
        elevation=280.0,
        region=NaturalRegion.ORINOQUIA,
        department="Arauca",
        municipality="Fuente De Oro",
    ),
    StationMetadata(
        code="34015010",
        name="LAS GAVIOTAS",
        latitude=4.553944,
        longitude=-70.930111,
        elevation=173.0,
        region=NaturalRegion.ORINOQUIA,
        department="Arauca",
        municipality="Cumaribo",
    ),
    StationMetadata(
        code="35225030",
        name="MODULOS - AUT",
        latitude=4.910472,
        longitude=-71.433056,
        elevation=139.0,
        region=NaturalRegion.ORINOQUIA,
        department="Meta",
        municipality="Orocué",
    ),
    StationMetadata(
        code="35215020",
        name="AEROPUERTO YOPAL - AUT",
        latitude=5.320444,
        longitude=-72.3875,
        elevation=314.0,
        region=NaturalRegion.ORINOQUIA,
        department="Casanare",
        municipality="Yopal",
    ),
    StationMetadata(
        code="38015030",
        name="AEROPUERTO PUERTO CARRENO",
        latitude=6.182436,
        longitude=-67.491222,
        elevation=55.0,
        region=NaturalRegion.ORINOQUIA,
        department="Vichada",
        municipality="Puerto Carreño",
    ),
    StationMetadata(
        code="36025010",
        name="TAME",
        latitude=6.456194,
        longitude=-71.745028,
        elevation=326.0,
        region=NaturalRegion.ORINOQUIA,
        department="Arauca",
        municipality="Tame",
    ),
    StationMetadata(
        code="37045010",
        name="SARAVENA - AUT",
        latitude=6.946389,
        longitude=-71.890556,
        elevation=251.0,
        region=NaturalRegion.ORINOQUIA,
        department="Arauca",
        municipality="Saravena",
    ),
    StationMetadata(
        code="37055010",
        name="AEROPUERTO SANTIAGO PÃ‰REZ",
        latitude=7.069444,
        longitude=-70.738056,
        elevation=131.0,
        region=NaturalRegion.ORINOQUIA,
        department="Arauca",
        municipality="Arauca",
    ),
    StationMetadata(
        code="57025020",
        name="GORGONA GUAPI",
        latitude=2.962944,
        longitude=-78.174361,
        elevation=7.0,
        region=NaturalRegion.INSULAR,
        department="Cauca",
        municipality="Guapi",
    ),
    StationMetadata(
        code="57015010",
        name="MALPELO - AUT",
        latitude=4.096,
        longitude=-81.608833,
        elevation=10.0,
        region=NaturalRegion.INSULAR,
        department="Valle del Cauca",
        municipality="Buenaventura",
    ),
    StationMetadata(
        code="14015060",
        name="ISLAS DEL ROSARIO",
        latitude=10.183333,
        longitude=-75.75,
        elevation=2.0,
        region=NaturalRegion.INSULAR,
        department="Bolívar",
        municipality="Cartagena De Indias",
    ),
    StationMetadata(
        code="17015010",
        name="AEROPUERTO SESQUICENTENARIO",
        latitude=12.542183,
        longitude=-81.730969,
        elevation=10.0,
        region=NaturalRegion.INSULAR,
        department="San Andrés y Providencia",
        municipality="San Andrés",
    ),
    StationMetadata(
        code="17025020",
        name="AEROPUERTO EL EMBRUJO",
        latitude=13.3595,
        longitude=-81.357722,
        elevation=2.0,
        region=NaturalRegion.INSULAR,
        department="San Andrés y Providencia",
        municipality="Providencia",
    ),
]


class StationCatalog:
    """Catalog managing metadata for Colombian meteorological stations."""

    def __init__(self, stations: Optional[List[StationMetadata]] = None) -> None:
        """Initialize catalog with benchmark stations or a custom list."""
        station_list = stations if stations is not None else BENCHMARK_STATIONS
        self._stations: Dict[str, StationMetadata] = {
            st.code: st for st in station_list
        }

    def get_by_code(self, code: str) -> StationMetadata:
        """Retrieve station metadata by its 8-digit IDEAM code."""
        if code not in self._stations:
            raise KeyError(f"Station code '{code}' not found in StationCatalog.")
        return self._stations[code]

    def get(
        self, code: str, default: Optional[StationMetadata] = None
    ) -> Optional[StationMetadata]:
        """Retrieve station metadata by code, or return default if not found."""
        return self._stations.get(code, default)

    def filter_by_region(self, region: NaturalRegion | str) -> List[StationMetadata]:
        """Filter stations belonging to a specific natural region."""
        target_region = NaturalRegion(region) if isinstance(region, str) else region
        return [st for st in self._stations.values() if st.region == target_region]

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Return global spatial bounding box (min_lat, max_lat, min_lon, max_lon)."""
        lats = [st.latitude for st in self._stations.values()]
        lons = [st.longitude for st in self._stations.values()]
        return (min(lats), max(lats), min(lons), max(lons))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert catalog to a pandas DataFrame."""
        records = [asdict(st) for st in self._stations.values()]
        df = pd.DataFrame(records)
        df["region"] = df["region"].apply(lambda r: r.value)
        return df

    def __len__(self) -> int:
        return len(self._stations)

    def __getitem__(self, code: str) -> StationMetadata:
        return self.get_by_code(code)

    def __iter__(self) -> Iterator[StationMetadata]:
        return iter(self._stations.values())
