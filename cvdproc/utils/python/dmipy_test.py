# # load the necessary modules
# from dmipy.core import modeling_framework
# from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
# from os.path import join
# import numpy as np
# from dmipy.signal_models import cylinder_models

# # the HCP acquisition parameters are saved in the following toolbox path:
# acquisition_path = modeling_framework.GRADIENT_TABLES_PATH

# # we can then load the parameters themselves and convert them to SI units:
# bvalues = np.loadtxt(join(acquisition_path, 'bvals_hcp_wu_minn.txt'))  # given in s/mm^2
# bvalues_SI = bvalues * 1e6  # now given in SI units as s/m^2
# gradient_directions = np.loadtxt(join(acquisition_path, 'bvecs_hcp_wu_minn.txt'))  # on the unit sphere

# # The delta and Delta times we know from the HCP documentation in seconds
# delta = 0.0106  
# Delta = 0.0431 

# # The acquisition scheme we use in the toolbox is then created as follows:
# acq_scheme = acquisition_scheme_from_bvalues(bvalues_SI, gradient_directions, delta, Delta)

# acq_scheme.print_acquisition_info

# stick = cylinder_models.C1Stick()

# from dmipy.core.modeling_framework import MultiCompartmentModel
# stick_model = MultiCompartmentModel(models=[stick])

# stick_model.parameter_cardinality

# mu = (np.pi / 2., np.pi / 2.)  # in radians
# lambda_par = 1.7e-9  # in m^2/s
# parameter_vector = stick_model.parameters_to_parameter_vector(
#     C1Stick_1_lambda_par=lambda_par, C1Stick_1_mu=mu)
# print(parameter_vector)

# E = stick_model.simulate_signal(acq_scheme, parameter_vector)

# res = stick_model.fit(acq_scheme, E)

# print('Optimized result:', res.fitted_parameters_vector)
# print('Ground truth:    ', parameter_vector)

# load the necessary modules
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
import numpy as np

acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()

stick = cylinder_models.C1Stick()
ball = gaussian_models.G1Ball()

ball_and_stick = MultiCompartmentModel(models=[ball, stick])

mu = (np.pi / 2., np.pi / 2.)
lambda_par = 1.7e-9
lambda_iso = 3e-9
partial_volume = 0.5  # we just give the same volume fraction for both fractions

parameter_vector = ball_and_stick.parameters_to_parameter_vector(
    C1Stick_1_lambda_par=lambda_par,
    G1Ball_1_lambda_iso=lambda_iso,
    C1Stick_1_mu=mu,
    partial_volume_0=partial_volume,
    partial_volume_1=partial_volume
)

E = ball_and_stick.simulate_signal(acq_scheme, parameter_vector)

fitted_ball_and_stick = ball_and_stick.fit(acq_scheme, E)