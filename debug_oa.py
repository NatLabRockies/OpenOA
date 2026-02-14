import pandas as pd
from openoa.plant import PlantData
from openoa.analysis.aep import MonteCarloAEP
import os
import traceback

def debug_oa():
    # Define the base data path
    data_path = "examples/data/la_haute_borne"

    print("Loading data...")
    try:
        # Load Asset Data
        asset = pd.read_csv(os.path.join(data_path, "la-haute-borne_asset_table.csv"))

        # Load SCADA Data
        scada = pd.read_csv(os.path.join(data_path, "la-haute-borne-data-2014-2015.csv"))
        scada["Date_time"] = pd.to_datetime(scada["Date_time"], utc=True).dt.tz_localize(None)

        # Load Meter Data (contains energy, availability, curtailment)
        meter = pd.read_csv(os.path.join(data_path, "plant_data.csv"))
        meter["time_utc"] = pd.to_datetime(meter["time_utc"], utc=True).dt.tz_localize(None)

        # Use same file for curtail data as it contains those columns
        curtail = meter.copy()

        # Load Reanalysis Data - ERA5 and MERRA2 have leading index columns
        era5 = pd.read_csv(os.path.join(data_path, "era5_wind_la_haute_borne.csv"), index_col=0)
        era5["datetime"] = pd.to_datetime(era5["datetime"], utc=True).dt.tz_localize(None)
        
        merra2 = pd.read_csv(os.path.join(data_path, "merra2_la_haute_borne.csv"), index_col=0)
        merra2["datetime"] = pd.to_datetime(merra2["datetime"], utc=True).dt.tz_localize(None)

        # Define Metadata
        metadata = {
            "capacity": 8.2,
            "asset": {
                "asset_id": "Wind_turbine_name",
                "latitude": "Latitude",
                "longitude": "Longitude"
            },
            "meter": {
                "time": "time_utc",
                "MMTR_SupWh": "net_energy_kwh"
            },
            "curtail": {
                "time": "time_utc",
                "IAVL_DnWh": "availability_kwh",
                "IAVL_ExtPwrDnWh": "curtailment_kwh"
            },
            "scada": {
                "time": "Date_time",
                "asset_id": "Wind_turbine_name",
                "WTUR_W": "P_avg",
                "WMET_HorWdSpd": "Ws_avg"
            },
            "reanalysis": {
                "era5": {
                    "time": "datetime",
                    "WMETR_HorWdSpd": "ws_100m",
                    "WMETR_HorWdSpdU": "u_100",
                    "WMETR_HorWdSpdV": "v_100",
                    "WMETR_AirDen": "dens_100m"
                },
                "merra2": {
                    "time": "datetime",
                    "WMETR_HorWdSpd": "ws_50m",
                    "WMETR_HorWdSpdU": "u_50",
                    "WMETR_HorWdSpdV": "v_50",
                    "WMETR_AirDen": "dens_50m"
                }
            }
        }

        print("Initializing PlantData...")
        plant = PlantData(
            metadata=metadata,
            scada=scada,
            meter=meter,
            curtail=curtail,
            asset=asset,
            reanalysis={"era5": era5, "merra2": merra2}
        )

        
        print("Running AEP Analysis...")
        pa = MonteCarloAEP(plant)
        pa.run(num_sim=10, progress_bar=False)
        
        print(f"AEP Result: {pa.results.mean()}")
        print("Analysis finished successfully!")

    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_oa()
