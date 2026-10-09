# Cartographic Boundary Assets for TROPICOR

This directory contains bundled vector boundaries for publication-quality, offline station mapping in Colombia and northwestern South America.

## Data Sources & Provenance

1. **Mainland Colombia & Neighboring Countries (50m scale)**:
   - Source: [Natural Earth](https://www.naturalearthdata.com/) - Public Domain (`ne_50m_admin_0_countries`).
   - Files:
     - `colombia_boundary_50m.geojson` (~19 KB): National boundary of Colombia.
     - `colombia_and_neighbors_50m.geojson` (~129 KB): Colombia plus Panama, Venezuela, Brazil, Peru, and Ecuador for regional spatial context.

2. **Insular Territories & Outlying Islands (10m scale)**:
   - Source: [Natural Earth](https://www.naturalearthdata.com/) - Public Domain (`ne_10m_admin_1_states_provinces`).
   - File:
     - `colombia_islands_10m.geojson` (~8.5 KB): High-resolution coastlines for:
       - **San Andrés y Providencia** (`COL-1342`): Archipelago in the Caribbean Sea (`~81.7°W - 81.3°W`, `~12.5°N - 13.4°N`).
       - **Isla de Malpelo** (`COL+99?`): Oceanic island sanctuary in the eastern Pacific Ocean (`~81.6°W`, `~3.98°N`).
   - *Rationale*: The standard 1:50m Natural Earth countries layer omits Malpelo and underrepresents San Andrés/Providencia as degenerate point fragments. Bundling this dedicated 8.5 KB 1:10m extraction ensures crisp island coastlines in map insets without requiring external GIS servers or multi-gigabyte shapefiles.

## License & Attribution

All Natural Earth vector datasets are in the **public domain**. Anyone may use, modify, and distribute these maps for any purpose without restriction.
