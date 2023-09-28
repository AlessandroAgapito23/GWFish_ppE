from GWFish.modules.fishermatrix import analyze_and_save_to_txt, compute_network_errors, compute_detector_fisher
import GWFish.modules.waveforms as waveforms
from GWFish.modules.detection import Network, Detector
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent


def test_gw190521_full_fisher(plot):
    df = pd.read_hdf(BASE_PATH / 'injections/GWTC3_cosmo_median.hdf5')
    
    params = df[df['event_ID']=='GW190521_074359']
    # do not perform the Fisher analysis over z
    z = params.pop('redshift')
    params['mass_1'] *= (1+z)
    params['mass_2'] *= (1+z)
    
    # the first parameter is the event ID
    fisher_params = params.columns.tolist()[1:]
    
    fisher_params = [
        # 'event_ID', 
        'mass_1', 
        'mass_2',
        'luminosity_distance', 
        'dec', 
        'ra',
        'theta_jn', 
        'psi', 
        'geocent_time',
        'phase', 
        # 'redshift', 
        'a_1',
        'a_2', 
        # 'tilt_1', 
        # 'tilt_2',
        # 'phi_12', 
        # 'phi_jl'
    ]
    
    network = Network(['LGWA'], detection_SNR=(0., 1.))
    
    network_snr, parameter_errors, sky_localization = compute_network_errors(
        network,
        params,
        fisher_parameters=fisher_params, 
        waveform_model='IMRPhenomXPHM'
    )
    
    fisher, square_snr = compute_detector_fisher(Detector('LGWA'), params.iloc[0], fisher_params, waveform_model='IMRPhenomXPHM')
    
    assert np.isclose(np.sqrt(square_snr), network_snr[0])

    if plot:
        plot_correlations(np.linalg.inv(fisher), fisher_params)


@pytest.mark.skip('Analysis results are not matching at the moment')
def test_fisher_analysis_output(mocker):
    params = {
        "mass_1": 1.4,
        "mass_2": 1.4,
        "redshift": 0.01,
        "luminosity_distance": 40,
        "theta_jn": 5 / 6 * np.pi,
        "ra": 3.45,
        "dec": -0.41,
        "psi": 1.6,
        "phase": 0,
        "geocent_time": 1187008882,
    }

    parameter_values = pd.DataFrame()
    for key, item in params.items():
        parameter_values[key] = np.full((1,), item)

    fisher_parameters = list(params.keys())

    network = Network(
        detector_ids=["ET"],
    )

    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
        waveform_class=waveforms.TaylorF2,
        waveform_model='TaylorF2',
    )

    header = (
        "network_SNR mass_1 mass_2 redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1 err_mass_2 "
        "err_redshift err_luminosity_distance err_theta_jn err_ra "
        "err_dec err_psi err_phase err_geocent_time err_sky_location"
    )

    data = [
        1.00000000000e02,
        1.39999999999e00,
        1.39999999999e00,
        1.00000000000e-02,
        4.00000000000e01,
        2.61799387799e00,
        3.45000000000e00,
        -4.09999999999e-01,
        1.60000000000e00,
        0.00000000000e00,
        1.18700888200e09,
        1.01791427671e-07,
        1.01791427689e-07,
        8.96883449508e-08,
        2.32204133549e00,
        1.04213847237e-01,
        3.12695677565e-03,
        2.69412953826e-03,
        2.04240222976e-01,
        4.09349000642e-01,
        5.63911212310e-05,
        2.42285325663e-05,
    ]

    assert np.savetxt.call_args.args[0] == "Errors_ET_test_SNR10.0.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }

@pytest.mark.skip('Analysis results are not matching at the moment')
def test_fisher_analysis_output_nosky(mocker):
    params = {
        "mass_1": 1.4,
        "mass_2": 1.4,
        "redshift": 0.01,
        "luminosity_distance": 40,
        "theta_jn": 5 / 6 * np.pi,
        "ra": 3.45,
        "dec": -0.41,
        "psi": 1.6,
        "phase": 0,
        "geocent_time": 1187008882,
    }

    parameter_values = pd.DataFrame()
    for key, item in params.items():
        parameter_values[key] = np.full((1,), item)

    fisher_parameters = list(params.keys())
    fisher_parameters.remove('dec')

    network = Network(
        detector_ids=["ET"],
    )


    mocker.patch("numpy.savetxt")

    analyze_and_save_to_txt(
        network=network,
        parameter_values=parameter_values,
        fisher_parameters=fisher_parameters,
        sub_network_ids_list=[[0]],
        population_name="test",
        waveform_class=waveforms.TaylorF2,
        waveform_model='TaylorF2',
    )

    header = (
        "network_SNR mass_1 mass_2 redshift luminosity_distance "
        "theta_jn ra dec psi phase geocent_time err_mass_1 err_mass_2 "
        "err_redshift err_luminosity_distance err_theta_jn err_ra "
        "err_psi err_phase err_geocent_time"
    )

    data = [
        100.0,
        1.400E+00,
        1.400E+00,
        1.000E-02,
        4.000E+01,
        2.618E+00,
        3.450E+00,
        -4.100E-01,
        1.600E+00,
        0.000E+00,
        1.187E+09,
        1.009E-07,
        1.009E-07,
        8.648E-08,
        2.321E+00,
        1.040E-01,
        3.121E-03,
        8.905E-02,
        1.771E-01,
        2.251E-05,
    ]

    assert np.savetxt.call_args.args[0] == "Errors_ET_test_SNR10.0.txt"
    assert np.allclose(np.savetxt.call_args.args[1], data, rtol=2e-3)

    assert np.savetxt.call_args.kwargs == {
        "delimiter": " ",
        "header": header,
        "comments": "",
        "fmt": (
            "%s %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E "
            "%.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E %.3E"
        ),
    }