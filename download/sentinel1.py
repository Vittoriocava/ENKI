"""
Download e processing Sentinel-1 SAR via STAC (Microsoft Planetary Computer).
Produce:
  - soil_state.tif (canale 5): stato attuale del suolo dal backscatter VV
  - flood_mask.tif (target): maschera binaria alluvione usando la "regola dello specchio"

Regola dello specchio (Specular Reflection Rule):
  L'acqua stagnante riflette il segnale SAR come uno specchio, producendo
  backscatter molto basso (pixel scuri). Confrontando una scena post-evento
  con un mosaico di riferimento mensile, le aree con forte calo di backscatter
  sono classificate come allagate.

  flood = (reference_mean_VV - event_VV) > threshold_dB

Fonte: Microsoft Planetary Computer — sentinel-1-grd collection
"""
import numpy as np
import rasterio
import rasterio.windows
from rasterio.merge import merge
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STATIC_DIR, EVENTS_DIR, GRID_SIZE, PIXEL_SIZE,
    SAR_POLARIZATION, SAR_INSTRUMENT_MODE, SAR_FLOOD_THRESHOLD_DB
)
from utils.geo import (
    compute_bbox_wgs84, compute_bbox_utm, get_target_profile,
    resample_array_to_grid
)


# ─────────────────────────────────────────────
# STAC Configuration
# ─────────────────────────────────────────────
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-grd"


def _init_stac():
    """Inizializza il client STAC con Planetary Computer signing."""
    try:
        import pystac_client
        import planetary_computer
        catalog = pystac_client.Client.open(
            STAC_API_URL,
            modifier=planetary_computer.sign_inplace,
        )
        return catalog
    except ImportError as e:
        raise ImportError(
            f"Librerie STAC non installate: {e}\n"
            "Installa con: pip install pystac-client planetary-computer"
        )


def search_s1_scenes(
    catalog,
    bbox_wgs: tuple,
    start_date: str,
    end_date: str,
    polarization: str = SAR_POLARIZATION,
) -> list:
    """
    Cerca scene Sentinel-1 GRD nel catalogo STAC.

    Args:
        catalog: client STAC inizializzato
        bbox_wgs: (lon_min, lat_min, lon_max, lat_max)
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        polarization: "VV" o "VH"

    Returns:
        lista di pystac Items
    """
    search = catalog.search(
        collections=[S1_COLLECTION],
        bbox=list(bbox_wgs),
        datetime=f"{start_date}/{end_date}",
    )

    items = list(search.items())

    # Filtra per polarizzazione disponibile
    return items


def _build_item_geo(item, src_width, src_height):
    """
    Costruisce il trasform affine WGS84 per un COG Sentinel-1 a partire
    dal bbox del STAC item e dalle dimensioni in pixel del raster.

    I COG S1 GRD su Planetary Computer NON hanno CRS / transform geo-referenziati:
      - src.crs = None
      - src.transform = identità (pixel coords)
    L'unica informazione geografica è item.bbox (WGS84).

    Returns:
        (affine_transform, CRS) in WGS84
    """
    from rasterio.crs import CRS as RioCRS
    from rasterio.transform import from_bounds

    lon_min, lat_min, lon_max, lat_max = item.bbox
    geo_transform = from_bounds(lon_min, lat_min, lon_max, lat_max,
                                src_width, src_height)
    geo_crs = RioCRS.from_epsg(4326)
    return geo_transform, geo_crs


def _compute_read_window(geo_transform, src_width, src_height, bbox_wgs):
    """
    Calcola la finestra di lettura usando il transform geo costruito dal bbox dell'item.

    Args:
        geo_transform: affine transform WGS84 costruito da item.bbox
        src_width: larghezza del COG in pixel
        src_height: altezza del COG in pixel
        bbox_wgs: (lon_min, lat_min, lon_max, lat_max) — AOI in WGS84

    Returns:
        rasterio.windows.Window o None se l'AOI è fuori dai bounds
    """
    from rasterio.windows import from_bounds

    lon_min, lat_min, lon_max, lat_max = bbox_wgs

    try:
        window = from_bounds(lon_min, lat_min, lon_max, lat_max, geo_transform)
        # Clip ai limiti del raster
        window = window.intersection(rasterio.windows.Window(
            0, 0, src_width, src_height
        ))
    except Exception:
        return None

    # Verifica dimensioni valide
    if window.width < 1 or window.height < 1:
        return None

    return window


def load_s1_composite(
    items: list,
    bbox_wgs: tuple,
    polarization: str = SAR_POLARIZATION,
    method: str = "median",
) -> np.ndarray:
    """
    Crea un composito (mediana) da multiple scene Sentinel-1.

    I COG S1 GRD su Planetary Computer non hanno CRS/transform geo:
    usiamo item.bbox per ricostruire il transform WGS84, poi windowed read
    dell'AOI e reproject a UTM 512×512 @40m.

    Args:
        items: lista di pystac Items
        bbox_wgs: bounding box WGS84 dell'AOI
        polarization: "VV" o "VH"
        method: "median" o "mean"

    Returns:
        np.ndarray (GRID_SIZE, GRID_SIZE) — backscatter in dB
    """
    import gc
    from rasterio.crs import CRS as RioCRS

    pol_key = polarization.lower()
    all_data = []
    geo_crs = RioCRS.from_epsg(4326)

    print(f"[SAR] Caricamento {len(items)} scene {polarization} (windowed COG)...")

    for i, item in enumerate(items):
        asset = item.assets.get(pol_key)
        if asset is None:
            continue

        try:
            with rasterio.open(asset.href) as src:
                # Costruisci transform geo dal bbox dell'item STAC
                geo_transform, _ = _build_item_geo(item, src.width, src.height)

                # Calcola la finestra per leggere solo l'AOI
                window = _compute_read_window(
                    geo_transform, src.width, src.height, bbox_wgs
                )

                if window is None:
                    print(f"[SAR]   Scena {i+1}/{len(items)}: "
                          f"skip — AOI fuori dai bounds della scena")
                    continue

                # Legge solo la finestra dell'AOI (molto meno RAM)
                data = src.read(1, window=window).astype(np.float32)

            # Calcola il transform geo per la finestra letta
            win_transform = rasterio.windows.transform(window, geo_transform)

            # Resample alla griglia target UTM
            resampled = resample_array_to_grid(
                data, win_transform, geo_crs, method="bilinear"
            )

            # Libera la memoria del dato grezzo
            del data
            gc.collect()

            # Converti a dB (S1 GRD da PC è in scala lineare)
            # backscatter_dB = 10 * log10(backscatter_linear)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resampled_db = np.where(
                    resampled > 0,
                    10.0 * np.log10(resampled),
                    np.nan  # Mantiene NoData come NaN per evitare artefatti
                )

            del resampled
            all_data.append(resampled_db)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_min = np.nanmin(resampled_db)
                r_max = np.nanmax(resampled_db)
            print(f"[SAR]   Scena {i+1}/{len(items)}: "
                  f"range=[{r_min:.1f}, {r_max:.1f}] dB")

        except Exception as e:
            print(f"[SAR]   Scena {i+1}/{len(items)}: errore — {e}")
            continue
        finally:
            gc.collect()

    if not all_data:
        raise RuntimeError("Nessuna scena SAR caricata con successo")

    # Stack e composito
    stack = np.stack(all_data, axis=0)
    del all_data
    gc.collect()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "median":
            composite = np.nanmedian(stack, axis=0)
        else:
            composite = np.nanmean(stack, axis=0)

    del stack
    gc.collect()

    return composite.astype(np.float32)


def download_sar_soil_state(
    event_date: datetime,
    output_dir: Path = None,
    use_stac: bool = True
) -> np.ndarray:
    """
    Scarica lo stato del suolo dal mosaico SAR mensile al tempo T0.

    Args:
        event_date: data dell'evento
        output_dir: directory di output
        use_stac: se True usa STAC, altrimenti dati sintetici

    Returns:
        np.ndarray (512, 512) — backscatter VV in dB
    """
    if output_dir is None:
        event_id = event_date.strftime("%Y%m%d")
        output_dir = EVENTS_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "soil_state.tif"

    if output_path.exists():
        print(f"[SAR] Soil state già esistente: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if use_stac:
        try:
            catalog = _init_stac()
            bbox_wgs = compute_bbox_wgs84()

            # Usa i 3 mesi precedenti come mosaico di baseline per lo stato del suolo pre-evento
            # (evita di includere l'alluvione nell'input del modello)
            start = (event_date - timedelta(days=90)).strftime("%Y-%m-%d")
            end = (event_date - timedelta(days=1)).strftime("%Y-%m-%d")

            print(f"[SAR] Ricerca scene S1 di baseline (soil state) per {start} → {end}...")
            items = search_s1_scenes(catalog, bbox_wgs, start, end)

            if not items:
                print(f"[SAR] Nessuna scena trovata per {start}→{end}, "
                      "provo range più ampio (6 mesi precedenti)...")
                start_ext = (event_date - timedelta(days=180)).strftime("%Y-%m-%d")
                items = search_s1_scenes(catalog, bbox_wgs, start_ext, end)

            if items:
                print(f"[SAR] Trovate {len(items)} scene, creazione composito...")
                # Limita a max 10 scene per velocità
                items = items[:10]
                composite = load_s1_composite(items, bbox_wgs)

                # Regola % NaN
                nan_ratio = np.isnan(composite).sum() / composite.size
                if nan_ratio > 0.10:
                    raise ValueError(f"Troppi pixel senza copertura SAR ({nan_ratio:.1%} NaN > 10%) nel soil state. Immagine scartata.")
                elif nan_ratio > 0:
                    print(f"[SAR] Trovati {nan_ratio:.1%} pixel NaN nel soil state. Interpolazione spaziale in corso...")
                    from rasterio.fill import fillnodata
                    valid_mask = ~np.isnan(composite)
                    composite_clean = np.where(valid_mask, composite, 0)
                    composite = fillnodata(composite_clean, mask=valid_mask, max_search_distance=100.0, smoothing_iterations=0)
                    
                    # Fallback
                    if np.isnan(composite).any():
                        composite[np.isnan(composite)] = np.nanmean(composite)

                composite_filled = composite.astype(np.float32)
                
                profile = get_target_profile()
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(composite_filled, 1)

                print(f"[SAR] Soil state salvato: {output_path}")
                return composite_filled
            else:
                raise ValueError("Nessuna scena scaricata da STAC disponibile, evento scartato.")

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Errore STAC: {e}")

    raise ValueError("Download STAC saltato o non disponibile, evento scartato (no dati sintetici).")


def download_sar_flood_mask(
    event_date: datetime,
    output_dir: Path = None,
    use_stac: bool = True
) -> np.ndarray:
    """
    Genera la maschera alluvione usando la Regola dello Specchio via STAC.

    Confronta il composito del mese dell'evento con il composito di
    riferimento (3 mesi precedenti).

    flood_mask = (reference_VV - event_VV) > threshold

    Args:
        event_date: data dell'evento di alluvione
        output_dir: directory di output
        use_stac: se True usa STAC

    Returns:
        np.ndarray (512, 512) — maschera binaria {0, 1}
    """
    if output_dir is None:
        event_id = event_date.strftime("%Y%m%d")
        output_dir = EVENTS_DIR / event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "flood_mask.tif"

    if output_path.exists():
        print(f"[SAR] Flood mask già esistente: {output_path}")
        with rasterio.open(output_path) as src:
            return src.read(1)

    if use_stac:
        try:
            catalog = _init_stac()
            bbox_wgs = compute_bbox_wgs84()

            # ── Scena del giorno dell'evento ──
            # Cerca il passaggio del satellite lo stesso giorno dell'alluvione o nei giorni immediatamente successivi
            start_event = event_date.strftime("%Y-%m-%d")
            end_event = (event_date + timedelta(days=1)).strftime("%Y-%m-%d")

            print(f"[SAR] Ricerca scena evento per {start_event} → {end_event}")
            event_items = search_s1_scenes(catalog, bbox_wgs, start_event, end_event)
            
            # Sentinel-1 passa ogni ~6 giorni, se non c'è il giorno dell'evento cerchiamo nei 5 giorni successivi
            if not event_items:
                raise ValueError("Nessuna scena scaricata da STAC disponibile, evento scartato.")

            # ── Composito di riferimento (3 mesi precedenti) ──
            ref_start = (event_date - timedelta(days=90)).strftime("%Y-%m-%d")
            ref_end = (event_date - timedelta(days=1)).strftime("%Y-%m-%d")

            print(f"[SAR] Composito riferimento: {ref_start} → {ref_end}")
            ref_items = search_s1_scenes(catalog, bbox_wgs, ref_start, ref_end)

            if event_items and ref_items:
                # Limita scene per velocità
                event_items = event_items[:8]
                ref_items = ref_items[:15]

                event_composite = load_s1_composite(event_items, bbox_wgs)
                ref_composite = load_s1_composite(ref_items, bbox_wgs)

                # Regola del 10% NaN
                nan_event = np.isnan(event_composite).sum() / event_composite.size
                nan_ref = np.isnan(ref_composite).sum() / ref_composite.size

                if nan_event > 0.10 or nan_ref > 0.10:
                    raise ValueError(f"Troppi NaN (evento: {nan_event:.1%}, baseline: {nan_ref:.1%} > 10%). Evento scartato.")

                # Se ci sono NaN (< 10%) nell'evento, li interpoliamo usando la baseline (ref_composite)
                if nan_event > 0:
                    print(f"[SAR] Interpolazione del {nan_event:.1%} di NaN nell'evento usando la baseline...")
                    mask_event_nan = np.isnan(event_composite)
                    event_composite[mask_event_nan] = ref_composite[mask_event_nan]

                # Se ci sono NaN nella baseline, facciamo il contrario per non perdere l'estensione
                if nan_ref > 0:
                    mask_ref_nan = np.isnan(ref_composite)
                    ref_composite[mask_ref_nan] = event_composite[mask_ref_nan]

                # Se entrambi sono still NaN nello stesso pixel, mettiamo a zero ma non causeranno falsi positivi
                both_nan = np.isnan(event_composite) & np.isnan(ref_composite)
                event_composite[both_nan] = 0
                ref_composite[both_nan] = 0

                # Regola dello specchio: maschera i NoData/NaN in uno dei due compositi
                with np.errstate(invalid='ignore'):
                    diff = ref_composite - event_composite
                    
                    flood_mask = ((diff > SAR_FLOOD_THRESHOLD_DB)).astype(np.float32)

                # Salva
                profile = get_target_profile()
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(flood_mask, 1)

                with rasterio.open(output_dir / "sar_event.tif", "w", **profile) as dst:
                    dst.write(event_composite.astype(np.float32), 1)

                with rasterio.open(output_dir / "sar_baseline.tif", "w", **profile) as dst:
                    dst.write(ref_composite.astype(np.float32), 1)

                flood_pct = 100 * flood_mask.sum() / flood_mask.size
                print(f"[SAR] Flood mask salvata: {output_path} — {flood_pct:.2f}% allagato")
                return flood_mask
            else:
                missing = []
                if not event_items:
                    missing.append("evento")
                if not ref_items:
                    missing.append("riferimento")
                raise ValueError(f"Scene STAC mancanti per: {', '.join(missing)}. Evento scartato.")

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Errore STAC flood mask: {e}")

    raise ValueError("Download STAC saltato o non disponibile per flood mask, evento scartato (no dati sintetici).")


if __name__ == "__main__":
    from datetime import datetime
    event = datetime(2023, 11, 15)
    soil = download_sar_soil_state(event, use_stac=True)
    flood = download_sar_flood_mask(event, use_stac=True)
    print(f"\nSoil state: shape={soil.shape}, range=[{soil.min():.1f}, {soil.max():.1f}] dB")
    print(f"Flood mask: shape={flood.shape}, {100*flood.mean():.2f}% allagato")
