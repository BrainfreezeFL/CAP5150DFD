import random


# This can also be considered one way injection. Injection method of 1 is client side and -1 is server side. 
def burst_observer(input, perturbation_rate, injection_method):
    output = list(input)
    previous_buffer_length = 2
    current_buffer_length = 0
    injection_buffer = injection_method
    pos_input = 0
    pos_output = 0
    array = [None,None]
    for i in input:
        
        if pos_input > 0 :
            if i == injection_method:
                if i == input[pos_input-1] and current_buffer_length == 1:
                    array = inject(output, perturbation_rate, injection_buffer, pos_output, previous_buffer_length)
                    pos_output = array[1] + pos_output
                    output = array[0]
                    current_buffer_length = current_buffer_length + 1
                    injection_buffer = i
                pos_input = pos_input + 1
                pos_output = pos_output + 1
                
            else :
                if input[pos_input-1] == injection_method:
                    previous_buffer_length = current_buffer_length
                current_buffer_length = 0
                pos_input = pos_input + 1
                pos_output = pos_output + 1
        elif i == injection_method:
            injection_buffer = i
            current_buffer_length = current_buffer_length + 1
            pos_input = pos_input + 1
            pos_output = pos_output + 1
        
        else : 
            current_buffer_length = 0
            pos_input = pos_input + 1
            pos_output = pos_output + 1
    print(output)
    return output
            
def inject(input, rate, what_to_inject, position, prev_buffer_size):
    
    array = [None,None]
    injection_amount = rate * prev_buffer_size
    x = 0
    while x < injection_amount :
    
        input.insert(position, what_to_inject)
        x = x + 1
        
    array[0] = input
    array[1] = injection_amount
    return array

# This performs two way injection by simply running the one way injection algorithm on the same string.
def two_way_injection(input, client_rate, server_rate) :
    output = burst_observer(input, client_rate, 1)
    output = burst_observer(output, server_rate, -1)
    
    return output
