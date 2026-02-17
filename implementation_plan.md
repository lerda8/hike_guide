# Advanced Mapy.cz Integration Plan

## Goal Description
Enhance the existing map integration to use official Mapy.cz APIs, providing a more premium experience for hikers. This includes official tile layers, elevation profiles, and improved route management.

## User Review Required
> [!IMPORTANT]
> **API Key**: You will need a Mapy.cz API key from [developer.mapy.com](https://developer.mapy.com/). I will provide a place in the app (or `.env` file) to configure this.
> **Credit Consumption**: The free tier provides 250,000 credits/month. Raster tiles and elevation lookups consume these credits.

## Proposed Changes

### Tech Stack Additions
-   **Frontend**: `Chart.js` for elevation profiles.
-   **Backend**: `httpx` for making asynchronous API calls to Mapy.cz REST services.

### Components

#### [MODIFY] Hike Detail View (`templates/hikes/detail.html`)
-   Update Leaflet config to use official Tile URLs with API key.
-   Add a collapsible "Elevation Profile" section with a Chart.js canvas.
-   Add proper Mapy.cz branding and attribution as per their terms.

#### [NEW] Elevation Service (`services/mapy_cz.py`)
-   Functions to call the Elevation API for a set of coordinates.
-   Cache results locally to save API credits.

#### [NEW] Route Import Enhancements
-   Add support for importing from Mapy.cz shared URLs (detecting `mapy.cz/s/...` and similar).
-   Use the Mapy.cz Routing API to "snap" tracks to known paths if requested.

#### [MODIFY] Configuration (`.env`)
-   Add `MAPY_CZ_API_KEY`.

## Verification Plan

### Automated Tests
-   Test `services/mapy_cz.py` with mock API responses.

### Manual Verification
1.  **Tile Loading**: Verify tiles load correctly and don't show "demo" watermarks once the API key is active.
2.  **Elevation Chart**: Upload a GPX and ensure the elevation chart appears and matches the terrain.
3.  **Link Import**: Paste a Mapy.cz shortcut link and verify the hike is created automatically.
