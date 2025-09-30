from datetime import UTC, datetime

from api_clients.neso import NESOClient


def main() -> None:
    neso_client = NESOClient()
    data = neso_client.get_interconnector_requirements(datetime(2025, 9, 23, tzinfo=UTC), datetime(2025, 9, 30, tzinfo=UTC))
    print(data)

if __name__ == "__main__":
    main()
