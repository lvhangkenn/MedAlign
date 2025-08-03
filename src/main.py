import os
import torch
from config import config
from utils import seed_everything, dill_load
from trainer import MedAlignTrainer

def run_single_model(config):
    print("Hello MedAlign!")
    print(config)

    if config['USE_CUDA']==True:
        device = torch.device("cuda:0"
                              "".format(config['GPU']))
        print(device)
        print(torch.tensor(0).to(device))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

        print(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        device = torch.device('cpu')
    torch.cuda.set_device(device)
    print(device)
    # data initial
    root="ready/"
    data_path = config['ROOT'] + root + "records_final.pkl"
    voc_path = config['ROOT'] + root + "voc_final.pkl"
    ddi_adj_path = config['ROOT'] + root + "ddi_A_final.pkl"
    ddi_mask_path = config['ROOT'] + root + "ddi_mask_H.pkl"
    molecule_path = config['ROOT'] + root + "atc3toSMILES.pkl"

    drug_smile_path = config['ROOT'] + root + "drug_smile.pkl"
    smile_subs_path = (config['ROOT'] + root + "smile_sub_b.pkl")
    smile_sub_voc_path = config['ROOT'] + root + "smile_sub_voc_b.pkl"
    smile_sub_degree_path = config['ROOT'] + root + "smile_sub_degree_b.pkl"
    smile_sub_recency_path = config['ROOT'] + root + "smile_sub_recency_b.pkl"
    drug_text_embs_path = config['ROOT'] + root + "drug_text_embs.pkl"


    ddi_adj, ddi_mask_H, data, molecule, voc = dill_load(ddi_adj_path), dill_load(ddi_mask_path), dill_load(data_path), dill_load(molecule_path), dill_load(voc_path)
    drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency ,drug_text_embs= dill_load(drug_smile_path), dill_load(smile_subs_path), dill_load(smile_sub_voc_path), dill_load(smile_sub_degree_path), dill_load(smile_sub_recency_path),dill_load(drug_text_embs_path)

    # model initial
    trainer = MedAlignTrainer(config, device, (ddi_adj, ddi_mask_H, data, molecule, voc), (drug_smile_matrix, smile_subs_matrix, smile_sub_voc, smile_sub_degree, smile_sub_recency),drug_text_embs)

    trainer.main()

    print("Everything is OK!")


if __name__ == '__main__':
    config = config
    seed_everything(config['SEED'])
    run_single_model(config)
