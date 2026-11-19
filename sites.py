"""Ground sites that get PARSEC parcel-model Nd.

Each entry points at the built hourly Nd file (the one the susceptibility
quicklooks read) and at the hourly size-distribution source on the common
10-1000 nm grid of pnsd_tools.GRID_NM. Paths are relative to the observation
repository (obs-ccn-nd-susceptibility, locally ARM_Data), passed to
build_parcel_inputs.py as --root or $OBS_REPO, so the same table works on the
laptop and on Arrhenius.

pnsd:  ("cache", <file in results/pnsd_cache>)    hourly dN/dlogDp already gridded
       ("csv",   <Virtanen et al. PNSD csv>)      gridded on the fly with pnsd_tools.hourly_pnsd_csv
kind:  "ndrop" (ARM NDROP retrieval), "cloudnet" (scaled-Frisch N), "insitu"

Sites built only on Arrhenius (Norunda, Melpitz, Bucharest, Maido, Andoya,
Oliktok, the Cloudnet versions of ENA/SGP/LASIC, Kenttarova and Ny-Alesund
if their caches are missing) are listed with the paths they will have once
their spectra are exported; build_parcel_inputs.py skips any site whose
files are not present and says so.
"""

SITES = {
    # --- ARM NDROP ground sites (SMPS) ---
    "ena":            dict(kind="ndrop", regime="Marine", built="results/ena/ena_2022_2026_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/ena_smps_pnsd_hourly.nc")),
    "sgp":            dict(kind="ndrop", regime="Continental", built="results/sgp/sgp_2016_2025_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/sgp_smps_pnsd_hourly.nc")),
    "epcape_pier":    dict(kind="ndrop", regime="Marine", built="results/epcape_pier/epc_2023_2024_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/epc_smps_pnsd_hourly.nc")),
    "lasic":          dict(kind="ndrop", regime="Marine", built="results/lasic/lasic_2016_2017_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/lasic_smps_pnsd_hourly.nc")),
    # --- ground in situ ---
    "puijo":          dict(kind="insitu", regime="Boreal", built="results/puijo_insitu/puijo_insitu.nc",
                           pnsd=("csv", "data/virtanen/Puijo_PNSD.csv")),
    "pallas":         dict(kind="insitu", regime="Boreal", built="results/pallas_insitu/pallas_insitu.nc",
                           pnsd=("csv", "data/virtanen/Pallas_PNSD.csv")),
    "zeppelin":       dict(kind="insitu", regime="Arctic", built="results/zeppelin_insitu/zeppelin_insitu.nc",
                           pnsd=("csv", "data/virtanen/Zeppelin_PNSD.csv")),
    "epcape_soledad": dict(kind="insitu", regime="Marine", built="results/epcape_soledad/epcape_soledad_hourly.nc",
                           pnsd=("cache", "results/pnsd_cache/epc_soledad_sems_pnsd_hourly.nc")),
    # --- Cloudnet sites ---
    "hyytiala":       dict(kind="cloudnet", regime="Boreal", built="results/hyytiala/hyytiala_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/hyytiala_dmps_pnsd_hourly.nc")),
    "mace-head":      dict(kind="cloudnet", regime="Marine", built="results/mace-head/mace-head_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/mace-head_smps_pnsd_hourly.nc")),
    "cabauw":         dict(kind="cloudnet", regime="Continental", built="results/cabauw/cabauw_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/cabauw_mpss_pnsd_hourly.nc")),
    "granada":        dict(kind="cloudnet", regime="Continental", built="results/granada/granada_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/granada_smps_pnsd_hourly.nc")),
    "palaiseau":      dict(kind="cloudnet", regime="Continental", built="results/palaiseau/palaiseau_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/palaiseau_smps_pnsd_hourly.nc")),
    "schneefernerhaus": dict(kind="cloudnet", regime="Continental",
                           built="results/schneefernerhaus/schneefernerhaus_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/schneefernerhaus_smps_pnsd_hourly.nc")),
    "kenttarova":     dict(kind="cloudnet", regime="Boreal", built="results/kenttarova/kenttarova_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/kenttarova_pnsd_hourly.nc")),
    "ny-alesund":     dict(kind="cloudnet", regime="Arctic", built="results/ny-alesund/ny-alesund_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/ny-alesund_pnsd_hourly.nc")),
    # --- built on Arrhenius: spectra still to be exported to results/pnsd_cache ---
    "norunda":        dict(kind="cloudnet", regime="Boreal", built="results/norunda/norunda_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/norunda_pnsd_hourly.nc")),
    "melpitz":        dict(kind="cloudnet", regime="Continental", built="results/melpitz/melpitz_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/melpitz_pnsd_hourly.nc")),
    "bucharest":      dict(kind="cloudnet", regime="Continental", built="results/bucharest/bucharest_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/bucharest_pnsd_hourly.nc")),
    "maido":          dict(kind="cloudnet", regime="Marine", built="results/maido/maido_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/maido_pnsd_hourly.nc")),
    "andoya":         dict(kind="cloudnet", regime="Arctic", built="results/andoya/andoya_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/andoya_pnsd_hourly.nc")),
    "oliktok":        dict(kind="cloudnet", regime="Arctic", built="results/oliktok/oliktok_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/oliktok_pnsd_hourly.nc")),
    "bankhead":       dict(kind="cloudnet", regime="Continental", built="results/bankhead/bankhead_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/bankhead_pnsd_hourly.nc")),
    "ena_cloudnet":   dict(kind="cloudnet", regime="Marine", built="results/ena/ena_cloudnet_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/ena_smps_pnsd_hourly.nc")),
    "sgp_cloudnet":   dict(kind="cloudnet", regime="Continental", built="results/sgp/sgp_cloudnet.nc",
                           pnsd=("cache", "results/pnsd_cache/sgp_smps_pnsd_hourly.nc")),
    "lasic_cloudnet": dict(kind="cloudnet", regime="Marine", built="results/lasic/lasic_cloudnet_smps.nc",
                           pnsd=("cache", "results/pnsd_cache/lasic_smps_pnsd_hourly.nc")),
}

# PARSEC mass-fraction order (8 columns in the Stockholm fork):
# h2so4, (nh4)hso4, (nh4)2so4, OC, BC, dust, seasalt, SVOC (unified-organics column; kept 0)
SPECIES = ["h2so4", "nh4hso4", "nh42so4", "oc", "bc", "dust", "seasalt", "svoc"]

# PLACEHOLDER compositions by regime, used only until the CAMS / ACSM stage
# overwrites them. Flagged comp_source = "regime_placeholder" in the inputs file.
REGIME_COMPOSITION = {
    "Marine":      [0, 0, 0.45, 0.20, 0.01, 0.00, 0.34, 0],
    "Boreal":      [0, 0, 0.30, 0.65, 0.05, 0.00, 0.00, 0],
    "Arctic":      [0, 0, 0.55, 0.30, 0.02, 0.00, 0.13, 0],
    "Continental": [0, 0, 0.40, 0.50, 0.10, 0.00, 0.00, 0],
}

# PLACEHOLDER cloud-base thermodynamics until the sounding / reanalysis stage.
PLACEHOLDER_T0_K = 280.0
PLACEHOLDER_P0_PA = 90000.0
