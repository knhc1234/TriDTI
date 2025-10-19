from utils.data import *
from utils.earlystopping import *
from models.model import *
#from models.protein_llm_feature import *
#from models.drug_llm_feature import *

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve
import gc

def show_result(data_path, AUC_list, AUPR_list, F1_list, Accuracy_list, Precision_list, Recall_list):
    AUC_mean, AUC_std = np.mean(AUC_list), np.std(AUC_list)
    AUPR_mean, AUPR_std = np.mean(AUPR_list), np.std(AUPR_list)
    F1_mean, F1_std = np.mean(F1_list), np.std(F1_list)
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_list), np.std(Accuracy_list)
    Precision_mean, Precision_std = np.mean(Precision_list), np.std(Precision_list)
    Recall_mean, Recall_std = np.mean(Recall_list), np.std(Recall_list)
    
    print("The model's results:")
    with open("{}/result/TriDTI_results.txt".format(data_path), 'w') as f:
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('AUPR(std):{:.4f}({:.4f})'.format(AUPR_mean, AUPR_std) + '\n')
        f.write('F1 Score(std):{:.4f}({:.4f})'.format(F1_mean, F1_std) + '\n')
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_std) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_std) + '\n')
        
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('AUPR(std):{:.4f}({:.4f})'.format(AUPR_mean, AUPR_std))
    print('F1 Score(std):{:.4f}({:.4f})'.format(F1_mean, F1_std))
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_std))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_std))

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Config
    print(dgl.__version__)
    print(torch.__version__)

    args = parse_arguments()
    config = load_config(args.config)
    
    DATASET_NAME = config["DATASET_NAME"]
    MAX_ATOM_NODES = config["MAX_ATOM_NODES"]
    MAX_DRUG_NODES = config["MAX_DRUG_NODES"]
    MAX_PROT_NODES = config["MAX_PROT_NODES"]

    top_k_d = config["top_k_d"]
    top_k_t = config["top_k_t"]
    hidden_dim = config["hidden_dim"]
    mol_dim = config["mol_dim"]
    prot_dim = config["prot_dim"]
    atom_dim = config["atom_dim"]
    graph_dim = config["graph_dim"]
    conv_dim = config["conv_dim"]
    proj_dim = config["proj_dim"]
    
    LR = config["LR"]
    EPOCHS = config["EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    pos_weight = config["pos_weight"]
    lambda_aux = 0.0001
    
    num_workers = 0

    print(f"Loaded config from {args.config}")
    print(f"Dataset: {DATASET_NAME}, LR: {LR}, Epochs: {EPOCHS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ['cpu', 'cuda']
    print(device)

    data_path = f"./dataset/{DATASET_NAME}"
    
    fold_auc_list = []
    fold_aupr_list = []
    fold_f1_list = []
    fold_acc_list = []
    fold_precision_list = []
    fold_recall_list = []
    
    for fold in range(5):
        result_path = f"./dataset/{DATASET_NAME}/result/fold{fold}"
        os.makedirs(result_path, exist_ok=True)
        
        print(f"Load Dataset: {DATASET_NAME}")
        train_dataloader, valid_dataloader, test_dataloader, test_df, drug_sim_graph, prot_sim_graph = get_dataloaders(data_path, fold, top_k_d, top_k_t, MAX_ATOM_NODES, MAX_DRUG_NODES, MAX_PROT_NODES, BATCH_SIZE, num_workers)
        
        model = TriDTI(
            mol_dim=mol_dim,
            prot_dim=prot_dim,
            hidden_dim=hidden_dim,
            gcn_dim = graph_dim,
            cnn_dim = conv_dim,
            projection_dim = proj_dim,
            num_heads=8,
        ).to(device)
        
        pos_weight = torch.FloatTensor([pos_weight]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss().to(device)
        best_auroc = 0
        
        es = EarlyStopping(patience = EPOCHS, verbose = True, delta=0, path=result_path + '/model.pt')
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader):
                molecular_graphs, drug_graphs, prot_graphs, d_ids, p_ids, dg_indices, pg_indices, drug_llms, protein_llms, prot_seqs, labels = batch
                
                batched_graph1 = [g.to(device) for g in molecular_graphs]
                batched_graph2 = [g.to(device) for g in drug_graphs]
                batched_graph3 = [g.to(device) for g in prot_graphs]
                
                d_id = d_ids.to(device)
                p_id = p_ids.to(device)
                
                drug_index = dg_indices.to(device)
                prot_index = pg_indices.to(device)
    
                drug_llm = drug_llms.to(device).float()
                protein_llm = protein_llms.to(device).float()
                prot_seq = prot_seqs.to(device)
                
                label = labels.to(device).float()
                    
                optimizer.zero_grad()
        
                out, aux_loss = model(batched_graph1, batched_graph2, batched_graph3, drug_llm, protein_llm, d_id, p_id, drug_index, prot_index, prot_seq)
                loss = criterion(out, label) + lambda_aux * aux_loss
                train_loss += loss.item()
                loss.backward()
                
                optimizer.step()
            clear_memory()
            
            print(f"Fold{fold} - Epoch : {epoch:4d}, Train Loss : {train_loss:.8f}") 
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                total_out = list()
                total_label = list()
                for batch in tqdm(valid_dataloader): 
                    molecular_graphs, drug_graphs, prot_graphs, d_ids, p_ids, dg_indices, pg_indices, drug_llms, protein_llms, prot_seqs, labels = batch
                
                    batched_graph1 = [g.to(device) for g in molecular_graphs]
                    batched_graph2 = [g.to(device) for g in drug_graphs]
                    batched_graph3 = [g.to(device) for g in prot_graphs]
                    
                    d_id = d_ids.to(device)
                    p_id = p_ids.to(device)
                    
                    drug_index = dg_indices.to(device)
                    prot_index = pg_indices.to(device)
        
                    drug_llm = drug_llms.to(device).float()
                    protein_llm = protein_llms.to(device).float()
                    prot_seq = prot_seqs.to(device)
                    
                    label = labels.to(device).float()
                    
                    out, aux_loss = model(batched_graph1, batched_graph2, batched_graph3, drug_llm, protein_llm, d_id, p_id, drug_index, prot_index, prot_seq)
                    
                    total_out.append(out)
                    total_label.append(label)
                    
                    loss = criterion(out, label) + lambda_aux * aux_loss
                    val_loss += loss.item()
                    
                clear_memory()
                
                print(f"Valid Loss : {val_loss:.8f}") 
                numpy_label_list = [tensor.cpu().numpy().flatten() for tensor in total_label]
                numpy_out_list = [tensor.cpu().numpy().flatten() for tensor in total_out]
                        
                all_labels = np.concatenate(numpy_label_list)
                all_out = np.concatenate(numpy_out_list)
                        
                auroc = roc_auc_score(all_labels, all_out)
                
                if best_auroc < auroc:
                    best_auroc = auroc
                    fold_best_label = all_labels
                    fold_best_out = all_out
                print("AUROC: ", auroc)
                
                es(auroc, model)
                if es.early_stop:
                    break
        
        del train_dataloader, valid_dataloader
        clear_memory()
        
        best_model = TriDTI(
            mol_dim=mol_dim,
            prot_dim=prot_dim,
            hidden_dim=hidden_dim,
            gcn_dim = graph_dim,
            cnn_dim = conv_dim,
            projection_dim = proj_dim,
            num_heads=8,
        ).to(device)
        
        best_model.load_state_dict(torch.load(result_path + '/model.pt'))
        best_model.eval()
        
        with torch.no_grad():
            total_out = list()
            total_label = list()
            for batch in tqdm(test_dataloader):
                molecular_graphs, drug_graphs, prot_graphs, d_ids, p_ids, dg_indices, pg_indices, drug_llms, protein_llms, prot_seqs, labels = batch
                
                batched_graph1 = [g.to(device) for g in molecular_graphs]
                batched_graph2 = [g.to(device) for g in drug_graphs]
                batched_graph3 = [g.to(device) for g in prot_graphs]
                    
                d_id = d_ids.to(device)
                p_id = p_ids.to(device)
                    
                drug_index = dg_indices.to(device)
                prot_index = pg_indices.to(device)
        
                drug_llm = drug_llms.to(device).float()
                protein_llm = protein_llms.to(device).float()
                prot_seq = prot_seqs.to(device)
                    
                label = labels.to(device).float()
                    
                out, aux_loss = best_model(batched_graph1, batched_graph2, batched_graph3, drug_llm, protein_llm, d_id, p_id, drug_index, prot_index, prot_seq)
                total_out.append(out)
                total_label.append(label)
                
            clear_memory()
            
            numpy_label_list = [tensor.cpu().numpy().flatten() for tensor in total_label]
            numpy_out_list = [tensor.cpu().numpy().flatten() for tensor in total_out]
            
            all_labels = np.concatenate(numpy_label_list)
            all_out = np.concatenate(numpy_out_list)
                         
            auroc = roc_auc_score(all_labels, all_out)
            auprc = average_precision_score(all_labels, all_out)
            
            precisions, recalls, thresholds = precision_recall_curve(all_labels, all_out)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # zero division
            optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]
            preds_binary = (all_out >= optimal_threshold).astype(int)
            
            f1 = f1_score(all_labels, preds_binary)
            acc = accuracy_score(all_labels, preds_binary)
            precision = precision_score(all_labels, preds_binary)
            recall = recall_score(all_labels, preds_binary)
            
            fold_auc_list.append(auroc)
            fold_aupr_list.append(auprc)
            fold_f1_list.append(f1)
            fold_acc_list.append(acc)
            fold_precision_list.append(precision)
            fold_recall_list.append(recall)
            
            print(f"Test AUROC: {auroc:.4f}")
            print(f"Test AUPRC: {auprc:.4f}")
            print(f"Test F1: {f1:.4f}")
            print(f"Test Accuracy: {acc:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            result_file_path = result_path + '/result.txt'
            with open(result_file_path, "w") as result_file:
                result_file.write("Drug\tTarget\tout\ttest_label\n")
                
                for i in range(len(test_df)):
                    drug = test_df["SMILES"].iloc[i]
                    target = test_df["Target Sequence"].iloc[i]
                    out_value = all_out[i]
                    label_value = all_labels[i]
                    result_file.write(f"{drug}\t{target}\t{out_value}\t{label_value}\n")
        
        del test_dataloader, test_df, drug_sim_graph, prot_sim_graph
        clear_memory()
    show_result(data_path, fold_auc_list, fold_aupr_list, fold_f1_list, fold_acc_list, fold_precision_list, fold_recall_list)

    