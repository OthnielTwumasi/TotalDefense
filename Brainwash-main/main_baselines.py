import sys, os, time
import numpy as np
import pickle as pkl    
import utils
import torch
from approaches.arguments import get_args
from resnet import ResNet18
# Keep defense utilities import, just remove adversarial_training from imports
from total_defense_util import (
    sanitize_data, validate_checkpoint, create_model_backup, 
    rollback_model, mitigate_brainwash_attack, store_checkpoint_hash
)

tstart = time.time()

def main(args):

    # Modify args parser in approaches/arguments.py to add these arguments
    # Or check them here if they were added elsewhere
    use_defense = getattr(args, 'use_defense', False)
    defense_sanitize = getattr(args, 'defense_sanitize', False)
    defense_validate = getattr(args, 'defense_validate', False)
    defense_backup = getattr(args, 'defense_backup', False)
    # Remove defense_adv_train flag
    defense_mitigate = getattr(args, 'defense_mitigate', False)

    if args.checkpoint != None:
        # Add checkpoint validation here
        if use_defense and defense_validate:
            print("Validating checkpoint integrity...")
            if not validate_checkpoint(args.checkpoint):
                print("Checkpoint validation failed!")
                if defense_backup:
                    if rollback_model(args.checkpoint):
                        print("Successfully rolled back to valid checkpoint.")
                    else:
                        print("Rollback failed. Proceeding with caution.")
                else:
                    print("No backup mechanism enabled. Proceeding with potentially corrupted checkpoint.")
        
        # Load checkpoint after validation
        checkpoint_dict = pkl.load(open(args.checkpoint, 'rb'))   

    if args.approach == 'afec_ewc' or args.approach == 'ewc' or args.approach == 'afec_rwalk' or args.approach == 'rwalk' or args.approach == 'afec_mas' or args.approach == 'mas' or args.approach == 'afec_si' or args.approach == 'si' or args.approach == 'ft' or args.approach == 'random_init' or args.approach == 'rwalk2':
        log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                        args.lamb, args.lr, args.batch_size, args.nepochs)
    elif args.approach == 'gs':
        log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_rho_{}_eta_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment,
                                                                                            args.approach, args.seed, 
                                                                                            args.lamb, args.mu, args.rho,
                                                                                                    args.eta,
                                                                                            args.lr, args.batch_size, args.nepochs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print('[CUDA unavailable]'); sys.exit()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Args -- Experiment
    if args.experiment == 'split_cifar100':
        # from dataloaders import split_cifar100 as dataloader
        from approaches.data_utils import generate_split_cifar100_tasks
    elif args.experiment == 'split_mini_imagenet':
        from approaches.data_utils import generate_split_mini_imagenet_tasks
    elif args.experiment == 'split_tiny_imagenet':
        from approaches.data_utils import generate_split_tiny_imagenet_tasks


    # Args -- Approach

    if args.approach == 'afec_ewc':
        from approaches import afec_ewc as approach
    elif args.approach == 'ewc':
        from approaches import ewc as approach
    elif args.approach == 'rwalk':
        from approaches import rwalk as approach
    elif args.approach == 'mas':
        from approaches import mas as approach


    print('Load data...')

    if args.experiment == 'split_cifar100':
        order = np.arange(100)
        im_sz = 32
        emb_fact = 1    
        data, taskcla, inputsize, task_order = generate_split_cifar100_tasks(args.tasknum, args.seed, rnd_order=False, order=order)
    elif args.experiment == 'split_mini_imagenet':
        order = np.arange(100)  
            
        class_num = 100 // (args.tasknum)  
        im_sz = 84
        emb_fact = 1    

        order = np.arange(100)
        home = os.path.expanduser('~')
        mini_root = os.path.join(home, 'data', 'miniImagenet' ) 
        data, taskcla, inputsize, task_order = generate_split_mini_imagenet_tasks(mini_root, task_num = args.tasknum, 
                                                                    rnd_order=False, order=order) 
        
    elif args.experiment == 'split_tiny_imagenet':
        order = np.arange(200)  
        home = os.path.expanduser('~')  
        root_add = os.path.join(home, 'data', 'tiny-imagenet-200') 
        dataset_file = './data/tiny_imagenet.npz'
        data, taskcla, inputsize, task_order = generate_split_tiny_imagenet_tasks(task_num = args.tasknum, 
                                                                    rnd_order=False, save_data=False,
                                                                    dataset_file=dataset_file, 
                                                                    order=order, root_add=root_add)
        
        class_num = 200 // (args.tasknum)  
        im_sz = 64
        emb_fact = 9
        
    # Add data sanitization defense here if needed
    if use_defense and defense_sanitize:
        print("Applying data sanitization defense...")
        for t in range(args.lasttask):
            # Create a temporary dataset-like object for sanitization
            class TempDataset:
                def __init__(self, x, y):
                    self.data = x
                    self.targets = y
                    
            # Sanitize train data
            temp_dataset = TempDataset(data[t]['train']['x'].cpu().numpy(), 
                                     data[t]['train']['y'].cpu().numpy())
            temp_dataset = sanitize_data(temp_dataset, threshold=2.0)
            data[t]['train']['x'] = torch.tensor(temp_dataset.data).cuda()
            data[t]['train']['y'] = torch.tensor(temp_dataset.targets).cuda()
            
            print(f"Task {t}: Sanitized training data shape: {data[t]['train']['x'].shape}")
            
    print('\nInput size =', inputsize, '\nTask info =', taskcla)


    ########################################################################################################################

    print('Inits...')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    nf = 32

    net = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()
    net_emp = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()


    ########################################################################################################################

    save_dict = {}  
    save_dict['scenario'] = args.scenario_name
    save_dict['model_type'] = 'resnet'    
    save_dict['dataset'] = args.experiment
    save_dict['class_num'] = data['ncla'] // args.tasknum  
    save_dict['bs'] = args.batch_size
    save_dict['lr'] = args.lr
    save_dict['n_epochs'] = args.nepochs
    save_dict['model'] = net.state_dict()
    save_dict['model_name'] = net.__class__.__name__
    save_dict['task_num'] = args.lasttask        
    save_dict['task_order'] = task_order
    save_dict['seed'] = args.seed    
    save_dict['emb_fact'] = emb_fact  
    save_dict['im_sz'] = inputsize[1]  

    cont_method_args = {'method': args.approach} 
    for tmp_key in args.__dict__.keys():    
        cont_method_args[tmp_key] = args.__dict__[tmp_key] 

    save_dict['cont_method_args'] = cont_method_args    

    if 'afec' in args.approach:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']  
            lamb_emp = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb_emp']  
        else:
            lamb = args.lamb    
            lamb_emp = args.lamb_emp    

        appr = approach.Appr(net, sbatch=args.batch_size, lamb=lamb, lamb_emp=lamb_emp, 
                            lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name,
                            empty_net = net_emp, clipgrad=args.clip)
    else:
        if args.checkpoint is not None:
            lamb = checkpoint_dict['pretrained_ckpt']['cont_method_args']['lamb']  
        else:
            lamb = args.lamb    

        appr = approach.Appr(net, lamb=lamb, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name, clipgrad=args.clip)


    if args.checkpoint is not None:
        # Store original model for comparison if using defense
        if use_defense and defense_mitigate:
            # Keep a copy of the pre-checkpoint model for comparison
            original_net = ResNet18(args.tasknum, data['ncla']//args.tasknum, nf=nf, include_head=True).cuda()
            original_net.load_state_dict(net.state_dict())
        
        appr.load_model(checkpoint_dict['pretrained_ckpt']['model'])   
        if 'afec' in args.approach:
            appr.load_emp_model(checkpoint_dict['pretrained_ckpt']['cont_method_args']['model_emp'])

        # If using defense, detect potential poisoning after loading checkpoint
        if use_defense and defense_mitigate:
            from total_defense_util import noise_detection
            print("Checking for potential poisoning in the loaded model...")
            if noise_detection(net, original_net, threshold=0.1):
                print("WARNING: Potential poisoning detected in the checkpoint model!")
                print("Applying mitigation techniques...")
                # Create a small dataloader for mitigation
                from torch.utils.data import TensorDataset, DataLoader
                # Use a sample of clean data for mitigation
                clean_x = []
                clean_y = []
                for t in range(args.lasttask):
                    # Take a small subset of clean data
                    idx = torch.randperm(len(data[t]['test']['x']), device='cuda')[:100]  # Sample 100 points
                    clean_x.append(data[t]['test']['x'][idx])
                    clean_y.append(data[t]['test']['y'][idx])
                
                clean_x = torch.cat(clean_x, dim=0)
                clean_y = torch.cat(clean_y, dim=0)
                clean_dataset = TensorDataset(clean_x, clean_y)
                clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=True)
                
                # Apply mitigation
                net = mitigate_brainwash_attack(net, clean_loader)
                print("Mitigation applied.")

        if args.init_acc:
            accs_tmp = []
            for u in range(checkpoint_dict['pretrained_ckpt']['task_num']):  
                xtest = data[u]['test']['x']
                ytest = data[u]['test']['y']
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                accs_tmp.append(test_acc *100)   
            
            with np.printoptions(precision=2, suppress=True):   
                print(np.array(accs_tmp) ) 
            
        

    print(appr.criterion)
    print('-' * 100)
    relevance_set = {}

    acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

    for t, ncla in taskcla:
        if args.checkpoint is not None and t < args.lasttask:
            print('Skip task {:2d} : {:15s}'.format(t, data[t]['name']))
            continue


        if t==1 and 'find_mu' in args.date:
            break

        if t == args.lasttask and args.checkpoint is None:  
            break
        
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(t, data[t]['name']))
        print('*' * 100)

        # Get data
        xtrain = data[t]['train']['x'].clone()
        xvalid = data[t]['test']['x'].clone()   

        ytrain = data[t]['train']['y'].clone()
        yvalid = data[t]['test']['y'].clone()

        # Add defense logic for detecting manipulated noise
        if args.checkpoint is not None and args.addnoise == True:
            # Before applying noise, check if we should sanitize it
            if use_defense and defense_sanitize:
                print("Checking potential poisoned noise...")
                
            if args.uniform is True:
                print('Using uniform noise')
                if 'inj_data_idx' not in checkpoint_dict.keys():
                    all_noise = torch.rand_like(xtrain) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']
                else:
                    print('Using uniform noise only on the injected data')
                    noise_prm = checkpoint_dict['rnd_idx_train']   
                    xtrain = xtrain[noise_prm]  
                    ytrain = ytrain[noise_prm]
                    all_noise = torch.zeros_like(xtrain)    
                    inj_idx = checkpoint_dict['inj_data_idx']   
                    print(f'number of noisy data : {len(inj_idx)}')
                    all_noise[inj_idx] = torch.rand_like(xtrain[inj_idx]) * 2 * checkpoint_dict['delta'] - checkpoint_dict['delta']

                xtrain = torch.clamp(xtrain + all_noise, 0, 1)  

                
            else:
                print('Using noise from checkpoint')
                all_noise = checkpoint_dict['latest_noise']   
                
                # If defense is enabled, analyze the noise
                if use_defense:
                    # Check if noise values seem suspicious (e.g., unusually high magnitude)
                    noise_magnitude = torch.norm(all_noise, p=2)
                    print(f"Noise magnitude: {noise_magnitude.item()}")
                    
                    if noise_magnitude > 5.0:  # Threshold can be adjusted
                        print("WARNING: Suspicious noise magnitude detected!")
                        if defense_mitigate:
                            print("Reducing noise magnitude...")
                            noise_scale_factor = 5.0 / noise_magnitude.item()
                            all_noise = all_noise * noise_scale_factor
                            print(f"Scaled noise magnitude: {torch.norm(all_noise, p=2).item()}")
                
                noise_prm = checkpoint_dict['rnd_idx_train']   
                xtrain = xtrain[noise_prm]
                ytrain = ytrain[noise_prm]
                xtrain = torch.clamp(xtrain + all_noise, 0, 1)  
        
        task = t

        # Train
        appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
        
        print('-' * 100)

        # Test
        for u in range(t + 1):
            xtest = data[u]['test']['x'].cuda()
            ytest = data[u]['test']['y'].cuda()
            test_loss, test_acc = appr.eval(u, xtest, ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                        100 * test_acc))
            acc[t, u] = test_acc
            lss[t, u] = test_loss
            
        # Save
        
        print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t,:t+1])))
        print('Save at ' + args.output)

        with np.printoptions(precision=2, suppress=True):   
            print(acc)

    if args.checkpoint is not None: 
        acc[:args.lasttask, :args.lasttask] = checkpoint_dict['pretrained_ckpt']['acc_mat'][:args.lasttask, :args.lasttask]
        
    # Done
    print('*' * 100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t', end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100 * acc[i, j]), end='')
        print()
    print('*' * 100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

    if args.checkpoint is not None: 
        acc_mat_sace_name = args.checkpoint.split('/')[-1]
        #remove the .pkl extension
        acc_mat_sace_name = acc_mat_sace_name[:-4]

        if args.addnoise is False:
            method = 'clean'
        if args.addnoise and args.uniform:  
            method = 'uniform'
        elif args.addnoise and args.uniform is False:
            method = 'ours'    
            
        # Add defense indicator to the saved filename
        if use_defense:
            method += '_defended'

        np.save(f'acc_mat_{acc_mat_sace_name}_{method}.npy', acc)   



    bwt_before = np.mean((acc[args.lasttask-1] - np.diag(acc))[:args.lasttask-1][:-1])
    avg_acc_before  = np.mean(acc[args.lasttask-1, :args.lasttask])

    if args.checkpoint is not None: 
        bwt_after = np.mean((acc[-1] - np.diag(acc))[:-1])
        avg_acc_after  = np.mean(acc[-1][:-1])
        last_task_acc = acc[-1, -1] 
        print(f'After BWT : {bwt_after} After avg acc : {avg_acc_after} Last task acc : {last_task_acc}')   

        
    print(f'Before BWT : {bwt_before} Before avg acc : {avg_acc_before}')  

    save_dict['last_task'] = int(args.lasttask)
    save_dict['acc_mat'] = acc
    save_dict['avg_acc'] = np.mean(acc[-1, :args.lasttask])
    save_dict['bwt'] = bwt_before
    save_dict['model'] = net.state_dict()
    if 'afec' in args.approach: 
        save_dict['cont_method_args']['model_emp'] = net_emp.state_dict()   

    # Add defense info to save_dict
    if use_defense:
        save_dict['defense_applied'] = True
        save_dict['defense_methods'] = {
            'sanitize': defense_sanitize,
            'validate': defense_validate,
            'backup': defense_backup,
            'mitigate': defense_mitigate
        }

    # Create backup of checkpoint if requested
    if use_defense and defense_backup:
        save_path = None
        save_name = f"task{args.tasknum}_epoch{args.nepochs}"
        if 'afec' not in args.approach:
            
            save_path = f'{args.approach}_lamb_{args.lamb}_{save_name}.pkl'
        else:
            save_path = f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_{save_name}.pkl'
            
        if os.path.exists(save_path):
            backup_path = create_model_backup(save_path)
            print(f"Created backup at {backup_path}")

    # save_dict['optim'] = optim.state_dict()

    save_name = utils.generate_save_name(save_dict)
    if args.checkpoint is None:
        #check if the file exists and add a number to the end if it does    
        if os.path.exists(f'{args.approach}_{save_name}.pkl'):
            print(f'File {args.approach}_{save_name}.pkl already exists. Saving with a different name.')
            if 'afec' not in args.approach:
                pkl_path = f'{args.approach}_lamb_{args.lamb}_{save_name}_1.pkl'
                pkl.dump(save_dict, open(pkl_path, 'wb'))
            else:
                pkl_path = f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_{save_name}_1.pkl'
                pkl.dump(save_dict, open(pkl_path, 'wb'))
        else:
            if 'afec' not in args.approach:
                pkl_path = f'{args.approach}_lamb_{args.lamb}_{save_name}.pkl'
                pkl.dump(save_dict, open(pkl_path, 'wb'))
            else:
                pkl_path = f'{args.approach}_lamb_{args.lamb}_lambemp_{args.lamb_emp}_{save_name}.pkl'
                pkl.dump(save_dict, open(pkl_path, 'wb'))
                
        # Store hash of the saved checkpoint
        if use_defense and defense_backup:
            print("Storing hash of newly saved checkpoint...")
            if 'pkl_path' in locals():
                store_checkpoint_hash(pkl_path)


if __name__ == '__main__':
    args = get_args()
    main(args)