# Source and third-party notices

The local development snapshot incorporates the two [upstream projects and pinned histories](docs/provenance/sources.json). The [migration manifest](docs/provenance/migration-manifest.tsv) records the corresponding paths and publication review status.

- **Europe source code.** The upstream snapshot states that a software license has yet to be declared. Obtain the rights holder's release decision before public distribution.
- **Gulf source code.** The original license notice describes academic research use with attribution and leaves data redistribution subject to the providers' permissions. [Gulf license](projects/gulf-market-anchored/LICENSE)
- **FootyStats.** Match statistics and derived inputs are subject to terms applicable to their acquisition and research use. [Site terms](https://footystats.org/terms-and-conditions)
- **Football-Data.co.uk.** The European integration code references the provider's opening and closing odds files. [Provider policy](https://football-data.co.uk/contact.php)
- **StatsBomb Open Data.** European event ingestion and momentum work reference StatsBomb. Follow the applicable use and attribution provisions. [Upstream data and license](https://github.com/hudl/open-data)
- **Python packages.** Each upstream `requirements.txt` is retained. Any downstream bundling must account for the licenses of installed versions.

See [Data and License Audit](docs/DATA_AND_LICENSE_AUDIT.md) and [Data Availability](docs/data-availability.md) for the asset-by-asset release boundaries.
