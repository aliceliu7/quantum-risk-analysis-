#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Based on IBM Qiskit Tutorial
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute

from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel as GCI
from qiskit.aqua.components.uncertainty_problems import UnivariatePiecewiseLinearObjective as PwlObjective
from qiskit.aqua.components.uncertainty_problems import MultivariateProblem
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.circuits  import FixedValueComparator as Comparator
from qiskit.aqua.algorithms import AmplitudeEstimation

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# define backend to be used
backend = BasicAer.get_backend('statevector_simulator')


# In[3]:


# set the model parameters
n_z = 2
z_max = 2
z_values = np.linspace(-z_max, z_max, 2**n_z)
p_zeros = [0.15, 0.25]
rhos = [0.1, 0.05]
lgd = [1, 2]
K = len(p_zeros)
alpha = 0.05


# In[4]:


# construct circuit factory for uncertainty model (Gaussian Conditional Independence model)
u = GCI(n_z, z_max, p_zeros, rhos)


# In[5]:


# determine the number of qubits required to represent the uncertainty model
num_qubits = u.num_target_qubits

# initialize quantum register and circuit
q = QuantumRegister(num_qubits, name='q')
qc = QuantumCircuit(q)

# construct circuit
u.build(qc, q)


# In[6]:


# run the circuit and analyze the results
job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))


# In[7]:


# analyze uncertainty circuit and determine exact solutions
p_z = np.zeros(2**n_z)
p_default = np.zeros(K)
values = []
probabilities = []
for i, a in enumerate(job.result().get_statevector()):
    
    # get binary representation
    b = ('{0:0%sb}' % num_qubits).format(i)
    prob = np.abs(a)**2

    # extract value of Z and corresponding probability    
    i_normal = int(b[-n_z:], 2)
    p_z[i_normal] += prob

    # determine overall default probability for k 
    loss = 0
    for k in range(K):
        if b[K - k - 1] == '1':
            p_default[k] += prob
            loss += lgd[k]
    values += [loss]
    probabilities += [prob]   

values = np.array(values)
probabilities = np.array(probabilities)
    
expected_loss = np.dot(values, probabilities)

losses = np.sort(np.unique(values))
pdf = np.zeros(len(losses))
for i, v in enumerate(losses):
    pdf[i] += sum(probabilities[values == v])
cdf = np.cumsum(pdf)

i_var = np.argmax(cdf >= 1-alpha)
exact_var = losses[i_var]
exact_cvar = np.dot(pdf[(i_var+1):], losses[(i_var+1):])/sum(pdf[(i_var+1):])


# In[8]:


# plot loss PDF, expected loss, var, and cvar
plt.bar(losses, pdf)
plt.axvline(expected_loss, color='green', linestyle='--', label='E[L]')
plt.axvline(exact_var, color='orange', linestyle='--', label='VaR(L)')
plt.axvline(exact_cvar, color='red', linestyle='--', label='CVaR(L)')
plt.legend(fontsize=15)
plt.xlabel('Loss L ($)', size=15)
plt.ylabel('probability (%)', size=15)
plt.title('Loss Distribution', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# plot results for Z
plt.plot(z_values, p_z, 'o-', linewidth=3, markersize=8)
plt.grid()
plt.xlabel('Z value', size=15)
plt.ylabel('probability (%)', size=15)
plt.title('Z Distribution', size=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

# plot results for default probabilities
plt.bar(range(K), p_default)
plt.xlabel('Asset', size=15)
plt.ylabel('probability (%)', size=15)
plt.title('Individual Default Probabilities', size=20)
plt.xticks(range(K), size=15)
plt.yticks(size=15)
plt.grid()
plt.show()

print('Expected Loss E[L]:                %.4f' % expected_loss)
print('Value at Risk VaR[L]:              %.4f' % exact_var)
print('P[L <= VaR[L]]:                    %.4f' % cdf[exact_var])
print('Conditional Value at Risk CVaR[L]: %.4f' % exact_cvar)


# In[9]:


# determine number of qubits required to represent total loss
n_s = WeightedSumOperator.get_required_sum_qubits(lgd)

# create circuit factory (add Z qubits with weight/loss 0)
agg = WeightedSumOperator(n_z + K, [0]*n_z + lgd)


# In[10]:


# define linear objective function
breakpoints = [0]
slopes = [1]
offsets = [0]
f_min = 0
f_max = sum(lgd)
c_approx = 0.25

objective = PwlObjective(
    agg.num_sum_qubits,
    0,
    2**agg.num_sum_qubits-1,  # max value that can be reached by the qubit register (will not always be reached)
    breakpoints, 
    slopes, 
    offsets, 
    f_min, 
    f_max, 
    c_approx
)

# define overall multivariate problem
multivariate = MultivariateProblem(u, agg, objective)


# In[11]:


num_qubits = multivariate.num_target_qubits
num_ancillas = multivariate.required_ancillas()

q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)

multivariate.build(qc, q, q_a)


# In[12]:


qc.draw()


# In[13]:


job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))


# In[14]:


# evaluate resulting statevector
value = 0
for i, a in enumerate(job.result().get_statevector()):
    b = ('{0:0%sb}' % multivariate.num_target_qubits).format(i)[-multivariate.num_target_qubits:]
    am = np.round(np.real(a), decimals=4)
    if np.abs(am) > 1e-6 and b[0] == '1':
        value += am**2

print('Exact Expected Loss:   %.4f' % expected_loss) 
print('Exact Operator Value:  %.4f' % value)
print('Mapped Operator value: %.4f' % multivariate.value_to_estimation(value))


# In[15]:


# running the  amplitude estimation
num_eval_qubits = 5
ae = AmplitudeEstimation(num_eval_qubits, multivariate)
result = ae.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))

print('Exact value:    \t%.4f' % expected_loss)
print('Estimated value:\t%.4f' % result['estimation'])
print('Probability:    \t%.4f' % result['max_probability'])


# In[16]:


# plot estimated values for "a"
plt.bar(result['values'], result['probabilities'], width=0.5/len(result['probabilities']))
plt.xticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('"a" Value', size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()

# plot estimated values for expected loss (after re-scaling and reversing the c_approx-transformation)
plt.bar(result['mapped_values'], result['probabilities'], width=1/len(result['probabilities']))
plt.axvline(expected_loss, color='red', linestyle='--', linewidth=2)
plt.xticks(size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('Expected Loss', size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()


# In[17]:


# define value x to evaluate the CDF(x)
def get_cdf_operator_factory(x_eval):

    # comparator as objective
    cdf_objective = Comparator(agg.num_sum_qubits, x_eval+1, geq=False)
    
    # define overall uncertainty problem
    multivariate_cdf = MultivariateProblem(u, agg, cdf_objective)
    
    return multivariate_cdf


# In[18]:


# set x value to estimate the CDF
x_eval = 2


# In[19]:


# get operator
multivariate_cdf = get_cdf_operator_factory(x_eval)

# get required number of qubits
num_qubits = multivariate_cdf.num_target_qubits
num_ancillas = multivariate_cdf.required_ancillas()  # TODO: why do we need two more ancillas?

# construct circuit
q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)

multivariate_cdf.build(qc, q, q_a)


# In[20]:


job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))


# In[21]:


qc.draw()


# In[22]:


# evaluate resulting statevector
var_prob = 0
for i, a in enumerate(job.result().get_statevector()):
    b = ('{0:0%sb}' % multivariate_cdf.num_target_qubits).format(i)[-multivariate_cdf.num_target_qubits:]
    prob = np.abs(a)**2
    if prob > 1e-6 and b[0] == '1':
        var_prob += prob
print('Operator CDF(%s)' % x_eval + ' = %.4f' % var_prob)
print('Exact    CDF(%s)' % x_eval + ' = %.4f' % cdf[x_eval])


# In[23]:


# run amplitude estimation
num_eval_qubits = 4
ae_cdf = AmplitudeEstimation(num_eval_qubits, multivariate_cdf)
# result_cdf = ae_cdf.run(quantum_instance=BasicAer.get_backend('qasm_simulator'), shots=100)
result_cdf = ae_cdf.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))


# In[24]:


# print results
print('Exact value:    \t%.4f' % cdf[x_eval])
print('Estimated value:\t%.4f' % result_cdf['estimation'])
print('Probability:    \t%.4f' % result_cdf['max_probability'])

# plot estimated values for "a"
plt.bar(result_cdf['values'], result_cdf['probabilities'], width=0.5/len(result['probabilities']))
plt.axvline(cdf[x_eval], color='red', linestyle='--', linewidth=2)
plt.xticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('CDF(%s)' % x_eval, size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()


# In[25]:


def run_ae_for_cdf(x_eval, num_eval_qubits=3, simulator='statevector_simulator'):
    
    # run amplitude estimation
    multivariate_var = get_cdf_operator_factory(x_eval)
    ae_var = AmplitudeEstimation(num_eval_qubits, multivariate_var)
    result_var = ae_var.run(BasicAer.get_backend(simulator))
    
    return result_var['estimation']


# In[26]:


def bisection_search(objective, target_value, low_level, high_level, low_value=None, high_value=None):
    """
    Determines the smallest level such that the objective value is still larger than the target
    :param objective: objective function
    :param target: target value
    :param low_level: lowest level to be considered
    :param high_level: highest level to be considered
    :param low_value: value of lowest level (will be evaluated if set to None)
    :param high_value: value of highest level (will be evaluated if set to None)
    :return: dictionary with level, value, num_eval
    """

    # check whether low and high values are given and evaluated them otherwise
    print('--------------------------------------------------------------------')
    print('start bisection search for target value %.3f' % target_value)
    print('--------------------------------------------------------------------')
    num_eval = 0
    if low_value is None:
        low_value = objective(low_level)
        num_eval += 1
    if high_value is None:
        high_value = objective(high_level)
        num_eval += 1    
        
    # check if low_value already satisfies the condition
    if low_value > target_value:
        return {'level': low_level, 'value': low_value, 'num_eval': num_eval, 'comment': 'returned low value'}
    elif low_value == target_value:
        return {'level': low_level, 'value': low_value, 'num_eval': num_eval, 'comment': 'success'}

    # check if high_value is above target
    if high_value < target_value:
        return {'level': high_level, 'value': high_value, 'num_eval': num_eval, 'comment': 'returned low value'}
    elif high_value == target_value:
        return {'level': high_level, 'value': high_value, 'num_eval': num_eval, 'comment': 'success'}

    # perform bisection search until
    print('low_level    low_value    level    value    high_level    high_value')
    print('--------------------------------------------------------------------')
    while high_level - low_level > 1:

        level = int(np.round((high_level + low_level) / 2.0))
        num_eval += 1
        value = objective(level)

        print('%2d           %.3f        %2d       %.3f    %2d            %.3f'               % (low_level, low_value, level, value, high_level, high_value))

        if value >= target_value:
            high_level = level
            high_value = value
        else:
            low_level = level
            low_value = value

    # return high value after bisection search
    print('--------------------------------------------------------------------')
    print('finished bisection search')
    print('--------------------------------------------------------------------')
    return {'level': high_level, 'value': high_value, 'num_eval': num_eval, 'comment': 'success'}


# In[27]:


# run bisection search to determine VaR
num_eval_qubits = 4
objective = lambda x: run_ae_for_cdf(x, num_eval_qubits=num_eval_qubits)
bisection_result = bisection_search(objective, 1-alpha, min(losses)-1, max(losses), low_value=0, high_value=1)
var = bisection_result['level']


# In[28]:


print('Estimated Value at Risk: %2d' % var)
print('Exact Value at Risk:     %2d' % exact_var)
print('Estimated Probability:    %.3f' % bisection_result['value'])
print('Exact Probability:        %.3f' % cdf[exact_var])


# In[29]:


# define linear objective
breakpoints = [0, var]
slopes = [0, 1]
offsets = [0, 0]  # subtract VaR and add it later to the estimate
f_min = 0
f_max = 3 - var
c_approx = 0.25

cvar_objective = PwlObjective(
    agg.num_sum_qubits,
    0,
    2**agg.num_sum_qubits-1,  # max value that can be reached by the qubit register (will not always be reached)
    breakpoints, 
    slopes, 
    offsets, 
    f_min, 
    f_max, 
    c_approx
)


# In[30]:


var = 2


# In[31]:


multivariate_cvar = MultivariateProblem(u, agg, cvar_objective)


# In[32]:


num_qubits = multivariate_cvar.num_target_qubits
num_ancillas = multivariate_cvar.required_ancillas()

q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)

multivariate_cvar.build(qc, q, q_a)


# In[33]:


job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))


# In[34]:


# evaluate resulting statevector
value = 0
for i, a in enumerate(job.result().get_statevector()):
    b = ('{0:0%sb}' % multivariate_cvar.num_target_qubits).format(i)[-multivariate_cvar.num_target_qubits:]
    am = np.round(np.real(a), decimals=4)
    if np.abs(am) > 1e-6 and b[0] == '1':
        value += am**2

# normalize and add VaR to estimate
value = multivariate_cvar.value_to_estimation(value)
normalized_value = value / (1.0 - bisection_result['value']) + var
print('Estimated CVaR: %.4f' % normalized_value)
print('Exact CVaR:     %.4f' % exact_cvar)


# In[ ]:


# run amplitude estimation
num_eval_qubits = 7
ae_cvar = AmplitudeEstimation(num_eval_qubits, multivariate_cvar)
# result_cvar = ae_cvar.run(quantum_instance=BasicAer.get_backend('qasm_simulator'), shots=100)
result_cvar = ae_cvar.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))


# In[ ]:


# print results
print('Exact CVaR:    \t%.4f' % exact_cvar)
print('Estimated CVaR:\t%.4f' % (result_cvar['estimation'] / (1.0 - bisection_result['value']) + var))
print('Probability:   \t%.4f' % result_cvar['max_probability'])


# In[ ]:


# plot estimated values for "a"
plt.bar(result_cvar['values'], result_cvar['probabilities'], width=0.5/len(result_cvar['probabilities']))
plt.xticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('"a" Value', size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()

# plot estimated values for expected loss (after re-scaling and reversing the c_approx-transformation)
normalized_values = np.array(result_cvar['mapped_values']) / (1.0 - bisection_result['value']) + var
plt.bar(normalized_values, result_cvar['probabilities'])
plt.axvline(exact_cvar, color='red', linestyle='--', linewidth=2)
plt.xticks(size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('CvaR', size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()


# In[ ]:




