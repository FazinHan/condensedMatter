import os

num = 1

while os.path.isfile(f'./output data/g_vs_eta_L{num}.npz'):
    num += 1

print(f'requesting ./output data/g_vs_eta_L{num}.npz from shivay...')

os.system(f'scp -P 4422 "fizaank.phy21.iitbhu@paramshivay.iitbhu.ac.in:/home/fizaank.phy21.iitbhu/massless_fermion/output_data/g_vs_eta_L{num}.npz" .')