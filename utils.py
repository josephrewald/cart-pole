import matplotlib.pyplot as plt
import torch
from experience import Experience

def plot(values, moving_avg_period, config):
    title = 'batch_size: ' + str(config['batch_size']) + \
            '; gamma: ' + str(config['gamma']) + \
            '\n eps_decay: ' + str(config['eps_decay']) + \
            '; target_update: ' + str(config['target_update']) + \
            '; lr: ' + str(config['lr'])
    filename = '/home/joseph/projects/deeplizard/rl/cart-pole/output/tune/' + \
               'bs-' + str(config['batch_size']) + \
               '_g-' + str(config['gamma']) + \
               '_ed-' + str(config['eps_decay']) + \
               '_tu-' + str(config['target_update']) + \
               '_lr-' + str(config['lr']) + \
               '.png'

    plt.figure(2)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.savefig(filename)
    #plt.pause(0.001)
    #print("Episode", len(values), "\n", \
    #    moving_avg_period, "episode moving avg:", moving_avg[-1])
    #if is_ipython: display.clear_output(wait=True)

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)
