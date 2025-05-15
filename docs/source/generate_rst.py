import os
import inspect
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))
import aurel.core as core

# Directory to save the generated .rst files
output_dir = "."

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File to write the documentation
output_file = os.path.join(output_dir, "source/core.rst")

gamma = ['gammadown3', 'gammaup3', 'dtgammaup3', 
         'gammadet', 'gammadown4', 'gammaup4']
extcurv = ['Kdown3', 'Kup3', 'Ktrace', 'Adown3', 
           'Aup3', 'A2']
lapse = ['alpha', 'dtalpha']
shift = ['betaup3', 'dtbetaup3', 'betadown3', 'betamag']
enne = ['nup4', 'ndown4']
gee = ['gdown4', 'gup4', 'gdet']
nullrayexp = ['null_ray_exp']
matter = ['press', 'eps', 'rho', 'rho_fromHam', 'enthalpy']
vel = ['w_lorentz', 'velup3', 'uup0', 'uup3', 'uup4', 
       'udown3', 'udown4', 'hdown4', 'hmixed4', 'hup4']
est = ['Tdown4']
fluid = ['rho_n', 'fluxup3_n', 'fluxdown3_n', 'angmomup3_n', 
         'angmomdown3_n', 'Stressup3_n', 
         'Stressdown3_n', 'Stresstrace_n', 'press_n', 
         'anisotropic_press_down3_n']
conserv = ['conserved_D', 'conserved_E', 'conserved_Sdown4', 
           'conserved_Sdown3', 'conserved_Sup4', 
           'conserved_Sup3', 'dtconserved']
kinema = ['thetadown4', 'theta', 'sheardown4', 'shear2']
s_curv = ['s_RicciS_u', 's_Gamma_udd3', 's_Riemann_uddd3', 
    's_Riemann_down3', 's_Ricci_down3', 's_RicciS']
st_curv = ['st_Gamma_udd4', 'st_Riemann_uddd4',
    'st_Riemann_down4', 'st_Riemann_uudd4',
    'st_Ricci_down4', 'st_Ricci_down3',
    'st_RicciS', 'Kretschmann']
constraints = ['Hamiltonian', 'Hamiltonian_Escale',
    'Momentumup3', 'Momentum_Escale']
gravimag = ['st_Weyl_down4', 'Weyl_Psi', 'Weyl_invariants',
    'eweyl_u_down4', 'eweyl_n_down3', 'bweyl_u_down4',
    'bweyl_n_down3']
varsdone = []

def print_subsec(title, subsecvars, allfunctions, varsdone):
    if title != "":
        f.write(title+"\n")
        f.write("-"*len(title)+"\n\n")
    for name in list(core.descriptions.keys()):
        if ((name in allfunctions) 
            and (name in subsecvars) 
            and (name not in varsdone)):
            # Check if the function is in `allfunctions`
            f.write(f"**{name}**: {core.descriptions[name]}\n\n")
            varsdone.append(name)
    return varsdone

# Start writing the .rst file
with open(output_file, "w") as f:
    f.write("aurel.core\n")
    f.write("##########\n\n")
    f.write(".. automodule:: aurel.core\n")
    f.write("   :noindex:\n\n")

    f.write(".. _descriptions_list:\n\n")
    # Add a section for functions listed in `descriptions`
    f.write("descriptions\n")
    f.write("************\n\n")
    allfunctions = []
    for name, func in inspect.getmembers(core.AurelCore, inspect.isfunction):
        allfunctions.append(name)

    f.write(".. _required_quantities:\n\n")
    f.write("Required quantities\n")
    f.write("===================\n\n")
    for name in list(core.descriptions.keys()):
        if ((name not in allfunctions)
            and (name not in varsdone)):
            # Check if the function is in `allfunctions`
            f.write(f"**{name}**: {core.descriptions[name]}\n\n")
            varsdone.append(name)

    f.write(".. _assumed_quantities:\n\n")
    f.write("Assumed quantities\n")
    f.write("==================\n\n")
    f.write(r"$\Lambda = 0$, the Cosmological constant, to change this do **AurelCore.Lambda = ...** before running calculations"+"\n\n")
    f.write(r'**alpha**: $\alpha = 1$, the lapse, to change this do **AurelCore.data["alpha"] = ...** before running calculations'+"\n\n")
    f.write(r"**dtalpha**: $\partial_t \alpha = 0$, the time derivative of the lapse"+"\n\n")
    f.write(r"**betaup3**: $\beta^i = 0$, the shift vector with spatial indices up"+"\n\n")
    f.write(r"**dtbetaup3**: $\partial_t \beta^i = 0$, the time derivative of the shift vector with spatial indices up"+"\n\n")
    f.write(r"**press**: $p = 0$, the fluid pressure"+"\n\n")
    f.write(r"**eps**: $\epsilon = 0$, the fluid specific internal energy"+"\n\n")
    f.write(r"**w_lorentz**: $W = 1$, the Lorentz factor"+"\n\n")
    f.write(r"**velup3**: $v^i = 0$, the Eulerian fluid three velocity with spatial indices up"+"\n\n")

    f.write("Metric quantities\n")
    f.write("=================\n\n")
    print_subsec("Spatial metric", gamma, allfunctions, varsdone)
    print_subsec("Extrinsic curvature", extcurv, allfunctions, varsdone)
    print_subsec("Lapse", lapse, allfunctions, varsdone)
    print_subsec("Shift", shift, allfunctions, varsdone)
    print_subsec("Timeline normal vector", enne, allfunctions, varsdone)
    print_subsec("Spacetime metric", gee, allfunctions, varsdone)

    f.write("Matter quantities\n")
    f.write("=================\n\n")
    f.write(r"Eulerian observer follows $n^\mu$"+"\n\n")
    f.write(r"Lagrangian observer follows $u^\mu$"+"\n\n")
    print_subsec("Lagrangian matter variables", matter, allfunctions, varsdone)
    print_subsec("Fluid velocity", vel, allfunctions, varsdone)
    print_subsec("Energy-stress tensor", est, allfunctions, varsdone)
    print_subsec("Eulerian matter variables", fluid, allfunctions, varsdone)
    print_subsec("Conserved variables", conserv, allfunctions, varsdone)
    print_subsec("Kinematic variables", kinema, allfunctions, varsdone)

    f.write("Curvature quantities\n")
    f.write("====================\n\n")
    print_subsec("Spatial curvature", s_curv, allfunctions, varsdone)
    print_subsec("Spacetime curvature", st_curv, allfunctions, varsdone)
    print_subsec("Weyl decomposition", gravimag, allfunctions, varsdone)

    f.write("Null ray expansion\n")
    f.write("==================\n\n")
    print_subsec("", nullrayexp, allfunctions, varsdone)

    f.write("Constraints\n")
    f.write("===========\n\n")
    print_subsec("", constraints, allfunctions, varsdone)

    if len(varsdone) != len(core.descriptions):
        f.write("Miscellaneous\n")
        f.write("=============\n\n")
        f.write("Need to update `docs/source/source/generate_rst.py`\n\n")
        print_subsec("", core.descriptions, allfunctions, varsdone)

    # Add a section for other functions
    f.write("AurelCore\n")
    f.write("*********\n\n")
    f.write(".. autoclass:: aurel.core.AurelCore\n")
    f.write("   :show-inheritance:\n")
    f.write("   :members:\n")