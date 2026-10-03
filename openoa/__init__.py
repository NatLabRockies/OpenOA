# NOTE: When bumping version, please be sure to also update parameters in sphinx/conf.py.

from openoa.plant import PlantData


__version__ = "3.2"


def __attach_methods():
    from openoa.analysis.aep import create_MonteCarloAEP
    from openoa.analysis.wake_losses import create_WakeLosses
    from openoa.analysis.eya_gap_analysis import create_EYAGapAnalysis
    from openoa.analysis.yaw_misalignment import create_StaticYawMisalignment
    from openoa.analysis.electrical_losses import create_ElectricalLosses
    from openoa.analysis.turbine_long_term_gross_energy import create_TurbineLongTermGrossEnergy

    PlantData.MonteCarloAEP = create_MonteCarloAEP
    PlantData.WakeLosses = create_WakeLosses
    PlantData.EYAGapAnalysis = create_EYAGapAnalysis
    PlantData.ElectricalLosses = create_ElectricalLosses
    PlantData.StaticYawMisalignment = create_StaticYawMisalignment
    PlantData.TurbineLongTermGrossEnergy = create_TurbineLongTermGrossEnergy


__attach_methods()
