import os

num = 1

while os.path.isfile(f'./output data/g_vs_eta_L{num}.npz'):
    num += 1

print(f'sending ./g_vs_ets_L.py to shivay...')

os.system(f'scp -P 4422 ./g_vs_eta_L.py fizaank.phy21.iitbhu@paramshivay.iitbhu.ac.in:/home/fizaank.phy21.iitbhu/massless_fermion/')