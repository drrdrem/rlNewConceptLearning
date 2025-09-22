import os, sys, time
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torchtext import data


def train(model, train_iter, eval_iter=None, lr=.01, epochs=256, test_interval=100, save_best=False, early_stop=None, seed=0):
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    
    start_t = time.time()

    for epoch in range(1, epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
        
            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            
            optimizer.step()

            steps += 1
            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch.batch_size
            sys.stdout.write(
                '\rEpoch [{}] Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{}) Time elapse(min): {:.1f}'.format(epoch,
                                                                                                        steps, 
                                                                                                        loss.data, 
                                                                                                        accuracy,
                                                                                                        corrects,
                                                                                                        batch.batch_size,
                                                                                                        (time.time()-start_t)/60))
            if eval_iter:
                if steps % test_interval == 0:
                    dev_acc = eval(eval_iter, model)
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        last_step = steps
            

def eval(data_iter, model):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)  # batch first, index align
        
        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum')

        avg_loss += loss.data
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_field):
    assert isinstance(text, str)
    model.eval()
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    output = model(x)    
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.data[0]+1]


def predict_prob(text, model, text_field, label_field):
    assert isinstance(text, str)
    model.eval()
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    output = model(x)
    prob = F.softmax(output, dim=1).data[0].tolist()
    return prob[1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def run(model, model_dir, drugs, alllab_df, allfea_df, fields):
    text_field = fields[0][1]
    label_field = fields[1][1]
    V = len(text_field.vocab.stoi) # number of vocabulary
    C = len(label_field.vocab.stoi)-1 # number of classes
    n_cnn = model(V=V, C=C)
    n_cnn.load_state_dict(torch.load(model_dir))

    read_papers = {}
    first_papers = {}
    for drug in drugs:
        print('########################' )
        print('#### drug {}'.format(drug))
        examples = []

        # filter all papers contain the drug.
        n_df = alllab_df[(alllab_df['drug']==drug) & (alllab_df['lab']!=2)].reset_index(drop=True)
        if len(n_df)==0:
            print("\nNo drug [{}]".format(drug))
            continue
        # Get predict reading prob of papers
        prob_list = []
        for idx in tqdm(range(len(n_df))):
            sample = n_df.loc[idx]

            drug, file, label = sample[0], sample[1], sample[2]

            example = allfea_df[allfea_df['file']==file].values.tolist()
            title, abstract = example[0][1], example[0][2]

            text = str(title)+" "+str(abstract)+" "+str(drug)+ " gene"
            prob = predict_prob(text, n_cnn, text_field, label_field)

            prob_list.append(prob)

        # Sort the prob.
        pro_idx = sorted(range(len(prob_list)), key=lambda k: prob_list[k])
        print('The first paper is {}'.format(n_df.loc[pro_idx[-1]]['file']))
        first_papers[drug] = n_df.loc[pro_idx[-1]]['file']

        read_list = []
        for idx in pro_idx[::-1]:
            sample = n_df.loc[idx]

            drug, file, label = sample['drug'], sample['file'], sample['lab']
            read_list.append(file)

            example = allfea_df[allfea_df['file']==file].values.tolist()
            title, abstract = example[0][1], example[0][2]

            text = str(title)+" "+str(abstract)+" "+str(drug)
            examples.append(data.Example.fromlist([text, label], fields))

            if label==1: break # if find an answer, stop
        print("Number of papers be read of drug [{}]: {}".format(drug, len(read_list)))
        read_papers[drug] = len(read_list)
        
    return read_papers, first_papers