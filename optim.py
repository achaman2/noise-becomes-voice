import numpy as np
import scipy.io as sio
from scipy.signal import correlate
from scipy.signal import convolve




def construct_Aty_regularized(A, y, L_x, num_speakers, num_listeners, N, N_p):

    #  The length of Hty is given by L_x*num_speakers 
    # y contains the set of signals intended to be transmitted of total length num_listeners*N, followed by a zero pad of total length num_speakers*N_p

    
    res_overall = np.zeros(num_speakers*L_x) 
    
    for i in range(num_listeners + num_speakers):
        
        A_current = A[i]
        res_current = np.zeros(num_speakers*L_x) 
        
        if i < num_listeners:
            
            y_current = y[i*N:(i+1)*N]
            
        else:
             y_current = y[num_listeners*N +(i-num_listeners)*N_p : num_listeners*N + (i-num_listeners + 1)*N_p]
             
        
        for j in range(num_speakers):
            
            h_i = A_current[j]
            
            corr = correlate(y_current, h_i)
            
            res_current[j*L_x:(j+1)*L_x] = corr[len(h_i)-1: L_x + len(h_i)-1]
            
        res_overall = res_overall+res_current
            
        
    return res_overall 


def construct_A_x_regularized(A, x, L_x, num_speakers, num_listeners, N, N_p):
    
    res_overall = []
    
    for i in range(num_listeners + num_speakers):
        
        if i < num_listeners:
            res = np.zeros(N)

        else:
            res = np.zeros(N_p)
        
        for j in range(num_speakers):

            h_i = A[i][j]
            res = res + convolve(h_i, x[j*L_x : (j+1)*L_x])

        res_overall.append(res)
        
    res_overall = np.concatenate(res_overall, axis = 0)
    
    return res_overall
    
    
    

def construct_A_x(A, x, L_x, N):
    
    num_listeners = len(A)
    num_speakers = len(A[0])
    
    res_overall = []
    
    for i in range(num_listeners):
        res = np.zeros(N)
        
        for j in range(num_speakers):

            h_i = A[i][j]
            
            res = res + convolve(h_i, x[j*L_x : (j+1)*L_x])
            
        res_overall.append(res)
        
    res_overall = np.concatenate(res_overall, axis = 0)
    
    return res_overall



def construct_Aty(A, y, L_x, N):

    #  The length of Aty is given by L_x*num_speakers 
    
    num_listeners = len(A)
    num_speakers = len(A[0])
    
    res_overall = np.zeros(num_speakers*L_x) 
    
    for i in range(num_listeners):
        
        A_current = A[i]
        res_current = np.zeros(num_speakers*L_x) 
        
            
        y_current = y[i*N:(i+1)*N]
            
        
        for j in range(num_speakers):
            
            h_i = A_current[j]
            
            corr = correlate(y_current, h_i)
            
            res_current[j*L_x:(j+1)*L_x] = corr[len(h_i)-1: L_x + len(h_i)-1]
            
        res_overall = res_overall+res_current
            
        
    return res_overall 




def project_y_on_null_space_of_A(y, A, L_x, N, num_iter = 100 ):
#     first project y on A^T, by minimizing z1 = argmin_z||y-A^Tz||^2. Then null space projection is y - A^Tz1
    
    num_listeners = len(A)
    z_0 = np.zeros(num_listeners * N)
    
    Ay = construct_A_x(A, y, L_x, N)
    
    Atz = construct_Aty(A, z_0, L_x, N)
    AAtz = construct_A_x(A, Atz, L_x, N)
    
    residue_0 = Ay - AAtz
    p_0 = residue_0
    
    for k in range(1, num_iter):
        
        Atp =  construct_Aty(A, p_0, L_x, N)
        
        alpha = np.dot(residue_0, residue_0)/np.dot(Atp, Atp)
        
        z_1 = z_0 + alpha*p_0
        
        AAtp = construct_A_x(A, Atp, L_x, N)
        
        residue_1 = residue_0 - alpha*AAtp
        
        beta = np.dot(residue_1, residue_1)/np.dot(residue_0, residue_0)
        
        p_1 = residue_1 + beta*p_0
        
        p_0 = p_1
        residue_0 = residue_1
        z_0 = z_1
        
    
    null_space_projection = y - construct_Aty(A, z_1, L_x, N)
        
    return null_space_projection
    
    


    



    
def solve_y_eq_Ax_nullspace_regularized(y, H_matrix, L_x, N, lambda_v, num_iter = 100):
    
    num_listeners = len(H_matrix)
    num_speakers = len(H_matrix[0])
    
    pad_len = L_x
#     padding necessary to perform Tikhonov regularization efficiently
    
    y_padded = np.concatenate((y, np.zeros( num_speakers * pad_len ) ), axis=0)
    H_matrix_with_lambda = H_matrix.copy()
    
    for i in range(num_speakers):
        
        H_temp = np.zeros((num_speakers, 1))
        H_temp[i, :] = np.sqrt(lambda_v)
        
        H_matrix_with_lambda.append(list(H_temp))
        
    
#  .............   now conjugate gradient descent...........
        
    x_0 = np.zeros(num_speakers*L_x)
    
    optim_variables = {'A': H_matrix_with_lambda, 
                       'L_x': L_x,
                       'num_speakers':num_speakers, 
                       'num_listeners': num_listeners,
                       'N': N,
                       'N_p': pad_len}
                      
    
    
    Aty = construct_Aty_regularized(y = y_padded, **optim_variables )
    A_x0 = construct_A_x_regularized(x = x_0, **optim_variables)
    
    AtA_x0 = construct_Aty_regularized(y = A_x0, **optim_variables )
    
    residue_0 = Aty - AtA_x0
    p_0 = residue_0
    
    for k in range(1, num_iter):
        A_p = construct_A_x_regularized(x = p_0, **optim_variables)
        AtA_p = construct_Aty_regularized(y = A_p, **optim_variables )
        
        alpha = np.dot(residue_0, residue_0)/np.dot(A_p, A_p)
        
        x_1 = x_0 + alpha*p_0
        
        residue_1 = residue_0 - alpha*AtA_p
        
        beta = np.dot(residue_1, residue_1)/np.dot(residue_0, residue_0)
        
        p_1 = residue_1 + beta*p_0
        
        p_0 = p_1
        residue_0 = residue_1
        x_0 = x_1
        
    return x_1
    



def solve_y_eq_Ax_mccs_regularized(y, H_with_noise_matrix, Noise_matrix, L_g, L_n, N, lambda_v, num_iter = 100):
    
    num_listeners = len(H_with_noise_matrix)
    num_speakers = len(H_with_noise_matrix[0])
    
    L_x = L_g + L_n - 1
    
    pad_len = L_x
#     padding necessary to perform Tikhonov regularization efficiently
    
    y_padded = np.concatenate((y, np.zeros( num_speakers * pad_len ) ), axis=0)
    H_with_noise_matrix_and_lambda = H_with_noise_matrix.copy()
    
    for i in range(num_speakers):
        
        H_temp = np.zeros((num_speakers, L_n))
        H_temp[i, :] = np.sqrt(lambda_v)*Noise_matrix[i]
        
        H_with_noise_matrix_and_lambda.append(list(H_temp))
        
    
#  .............   now conjugate gradient descent...........
        
    x_0 = np.zeros(num_speakers*L_g)
    
    optim_variables = {'A': H_with_noise_matrix_and_lambda, 
                       'L_x': L_g,
                       'num_speakers':num_speakers, 
                       'num_listeners': num_listeners,
                       'N': N,
                       'N_p': pad_len}
                      
    
    
    Aty = construct_Aty_regularized(y = y_padded, **optim_variables )
    A_x0 = construct_A_x_regularized(x = x_0, **optim_variables)
    
    AtA_x0 = construct_Aty_regularized(y = A_x0, **optim_variables )
    
    residue_0 = Aty - AtA_x0
    p_0 = residue_0
    
    for k in range(1, num_iter):
        A_p = construct_A_x_regularized(x = p_0, **optim_variables)
        AtA_p = construct_Aty_regularized(y = A_p, **optim_variables )
        
        alpha = np.dot(residue_0, residue_0)/np.dot(A_p, A_p)
        
        x_1 = x_0 + alpha*p_0
        
        residue_1 = residue_0 - alpha*AtA_p
        
        beta = np.dot(residue_1, residue_1)/np.dot(residue_0, residue_0)
        
        p_1 = residue_1 + beta*p_0
        
        p_0 = p_1
        residue_0 = residue_1
        x_0 = x_1
        
    return x_1
    
    
    
    
    
    
    
    
    
    
    
    








# def solve_y_eq_Hg_mccs_regularized(y, H_with_noise_matrix, Noise_matrix, L_g, Ln, N, lambda_v):
    
#     num_listeners = len(H_with_noise_matrix)
#     num_speakers = len(H_with_noise_matrix[0])
    
#     L_x = L_g + L_n - 1
    
#     pad_len = L_x
# #     padding necessary to perform Tikhonov regularization efficiently
    
#     y_padded = np.concatenate((y, np.zeros( num_speakers * pad_len ) ), axis=0)
#     H_with_noise_matrix_and_lambda = H_with_noise_matrix.copy()
    
#     for i in range(num_speakers):
        
#         H_temp = np.zeros((num_speakers, L_n))
#         H_temp[i, :] = np.sqrt(lambda_v)*Noise_matrix[i]
        
#         H_with_noise_matrix_and_lambda.append(list(H_temp))
        
    
# #  .............   now conjugate gradient descent...........
        
    
#     g_1 = np.zeros(num_speakers*L_g)
    
    
    
    
    
    
    
    
    
    
    
# def construct_H_x_regularized(H, x, L_x, num_speakers, num_listeners, N, N_p):
    
#     res_overall = []
    
#     for i in range(num_listeners + num_speakers):
        
#         if i < num_listeners:
#             res = np.zeros(N)

#         else:
#             res = np.zeros(N_p)
        
#         for j in range(num_speakers):

#             h_i = H_res[i][j]
#             res = res + convolve(h_i, x[j*L_x : (j+1)*L_x])

#         res_overall.append(res)
        
#     res_overall = np.concatenate(res_overall, axis = 0)
    
#     return res_overall

        
    
                                
# def construct_Hty_regularized(H, y, L_x, num_speakers, num_listeners, N, N_p):

#     #  The length of Hty is given by L_x*num_speakers 
#     # y contains the set of signals intended to be transmitted of total length num_listeners*N, followed by a zero pad of total length num_speakers*N_p

    
#     res_overall = np.zeros(num_speakers*L_x) 
    
#     for i in range(num_listeners + num_speakers):
        
#         H_current = H[i]
#         res_current = np.zeros(num_speakers*L_x) 
        
        
#         if i < num_listeners:
            
#             y_current = y[i*N:(i+1)*N]
            
#         else:
#              y_current = y[num_listeners*N +(i-num_listeners)*N_p : num_listeners*N + (i-num_listeners + 1)*N_p]
             

#         for j in range(num_speakers):
            
#             h_i = H_current[j]
            
#             corr = correlate(y_current, h_i)
            
#             res_current[j*L_x:(j+1)*L_x] = corr[len(h_i)-1:len(h_i)-1+L_x]
            
        
#         res_overall = res_overall+res_current
            
        
        
#     return res_overall       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #     Hty =  construct_Hty_regularized(H_with_noise_matrix_and_lambda, y_padded, L_g, num_speakers, num_listeners, N, N_p = pad_len)
    
# #     H_g = construct_H_x_regularized(H_with_noise_matrix_and_lambda, x_1, L_g, num_speakers, num_listeners, N, N_p = L_x)
    
# #     residue = Hty - construct_Hty_regularized( H_with_noise_matrix_and_lambda, H_x, L_g, num_speakers, num_listeners, N, N_p = L_x)
    
# #     p=residue
# #     A_p_val = construct_H_x_regularized(H_with_noise_matrix_and_lambda, p, L_g, num_speakers, num_listeners, N, N_p = L_x)
    
# #     A_p_val_list = []
# #     A_p_val_list.append(A_p_val)

# #     p_list=[]
# #     p_list.append(p)

# #     alpha=np.inner(p,residue)/np.inner(A_p_val,A_p_val)
    
# #     for k in range(1,num_iter):
            
# #         x_1 = x_1 + alpha*p
        
# #         H_x = construct_H_x_regularized(H_with_noise_matrix_and_lambda, x_1, L_g, num_speakers, num_listeners, N, N_p = L_x)
        
# #         residue = Hty - construct_Hty_regularized( H_with_noise_matrix_and_lambda, H_x, L_g, num_speakers, num_listeners, N, N_p = L_x)
        
# #         A_rk=prod_Hres_g_multiple_regularized(H_res_sig_lambda,residue,L_g,N,L_g,K)
    
    

    
    
    
    

    
    
    
    
    
    
    
    
    