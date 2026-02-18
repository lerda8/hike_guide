import os

from services import mapy_cz


def main() -> None:
    api_key = os.getenv("MAPY_CZ_API_KEY")
    if not api_key:
        raise SystemExit("MAPY_CZ_API_KEY is not set")

    tile_config = mapy_cz.tile_layer_config()
    map_url = mapy_cz.build_mapy_url_showmap([14.4203523, 50.0313731])

    print("Mapy.com API key detected.")
    print("Tile URL template:", tile_config["url"])
    print("Mapy.com showmap:", map_url)


if __name__ == "__main__":
    main()
