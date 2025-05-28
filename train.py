import time
import os
import logging
import torch
from setting.dataLoader import get_loader
from setting.utils import compute_accuracy
from setting.utils import create_folder, random_seed_setting
from setting.options import opt
from model.MCAMamba import Net
import utility
import glob  

os.environ['OPENBLAS_NUM_THREADS'] = '1'  
os.environ['OMP_NUM_THREADS'] = '1'  
random_seed_setting(6)  

torch.backends.cudnn.deterministic = False  
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True  
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id  
GPU_NUMS = torch.cuda.device_count()  

if not hasattr(opt, 'save_path'):
    opt.save_path = './checkpoints/'
    print(f"Warning: 'save_path' not found in options, using default: {opt.save_path}")

save_path = create_folder(opt.save_path + opt.dataset)  # Create save path based on dataset name
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))  # Get current time as model identifier

log_dir = save_path + '/log/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(save_path + '/weight/'):
    os.makedirs(save_path + '/weight/')

model_dir = save_path + '/model/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

old_models = glob.glob(os.path.join(model_dir, '*'))
for old_model in old_models:
    os.remove(old_model)

logging.basicConfig(filename=log_dir + opt.dataset + current_time + 'log.log', 
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
                    level=logging.INFO,  
                    filemode='a',  
                    datefmt='%Y-%m-%d %I:%M:%S %p')  


logging.info(f'********************start train!********************')
logging.info(f'Config--epoch:{opt.epoch}; lr:{opt.lr}; batch_size:{opt.batchsize};')

train_loader, test_loader, trntst_loader, all_loader, train_num, val_num, trntst_num = get_loader(
    dataset=opt.dataset,  
    batchsize=opt.batchsize,  
    num_workers=opt.num_work,  
    useval=opt.useval,  
    pin_memory=True  
)

logging.info(f'Loading data, including {train_num} training images and {val_num} \
        validation images and {trntst_num} train_test images')


model = Net(opt.dataset).cuda()

optimizer = torch.optim.Adam(model.parameters(), opt.lr)


T_max = opt.epoch  
eta_min = 1e-6  

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=T_max, eta_min=eta_min
)

logging.info(f'Using cosine annealing learning rate scheduler: T_max={T_max}, eta_min={eta_min}')

criterion = torch.nn.CrossEntropyLoss().cuda()


def train(train_loader, model, optimizer, epoch, save_path):
    """
    Training function: Execute one epoch of model training

    Parameters:
        train_loader: Training data loader
        model: Model to train
        optimizer: Optimizer
        epoch: Current training epoch
        save_path: Path to save the model
    """
    model.train()  
    loss_all = 0  
    iteration = len(train_loader)  
    acc = 0  
    num = 0  

    for i, (hsi, Xdata, hsi_pca, gt, h, w) in enumerate(train_loader, start=1):
        optimizer.zero_grad()  

        hsi = hsi.cuda()  
        Xdata = Xdata.cuda()  
        hsi_pca = hsi_pca.cuda()  
        gt = gt.cuda()  

        _, outputs = model(hsi_pca.unsqueeze(1), Xdata)  

        gt_loss = criterion(outputs, gt)  
        loss = gt_loss

        loss.backward()  
        optimizer.step()  

        loss_all += loss.detach()  
        acc += compute_accuracy(outputs, gt) * len(gt)  
        num += len(gt)  

    loss_avg = loss_all / iteration
    acc_avg = acc / num

    logging.info(f'Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_train_avg: {loss_avg:.4f}, acc_avg:{acc_avg:.4f}')

    if (epoch == opt.epoch or epoch == opt.epoch // 2):
        torch.save(optimizer.state_dict(),
                   save_path + '/weight/' + current_time + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
        torch.save(model.state_dict(),
                   save_path + '/weight/' + current_time + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))


best_acc = opt.best_acc
best_epoch = opt.best_epoch


def test(val_loader, model, epoch, save_path):
    """
    Test function: Evaluate model performance on validation set

    Parameters:
        val_loader: Validation data loader
        model: Model to evaluate
        epoch: Current training epoch
        save_path: Path to save the model
    """
    global best_acc, best_epoch  

    oa = 0.0

    if (opt.dataset == 'Berlin'):
        oa, aa, kappa, acc = utility.createBerlinReport(
            net=model,  
            data=val_loader,  
            device='cuda:0'  
        )
    # Evaluation for Houston2018 dataset
    elif (opt.dataset == 'Houston2018'):
        houston_class_names = ['Healthy Grass', 'Stressed Grass', 'Artificial Turf', 'Evergreen Trees',
                               'Deciduous Trees', 'Bare Earth', 'Water', 'Residential Buildings',
                               'Non-Residential Buildings', 'Roads', 'Sidewalks', 'Crosswalks',
                               'Major Thoroughfares', 'Highways', 'Railways', 'Paved Parking Lots',
                               'Unpaved Parking Lots', 'Cars', 'Trains', 'Stadium Seats']

        oa, aa, kappa, acc = utility.createReport(
            net=model,  
            data=val_loader,  
            class_names=houston_class_names,  
            device='cuda:0'  
        )
    elif (opt.dataset == 'Houston2013'):
        houston2013_class_names = ['Healthy Grass', 'Stressed Grass', 'Synthetic Grass', 'Trees',
                                   'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                                   'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                                   'Running Track']

        oa, aa, kappa, acc = utility.createReport(
            net=model,  
            data=val_loader,  
            class_names=houston2013_class_names,  
            device='cuda:0'  
        )
    elif (opt.dataset == 'Augsburg'):
        augsburg_class_names = ['Impervious Surface', 'Building', 'Low Vegetation',
                                'Tree', 'Car', 'Background/Clutter', 'Natural Ground']

        oa, aa, kappa, acc = utility.createReport(
            net=model,  
            data=val_loader,  
            class_names=augsburg_class_names,  
            device='cuda:0'  
        )
    else:
        print(f"Warning: No evaluation implementation for dataset '{opt.dataset}'")
        oa = 0.0
        aa = 0.0
        kappa = 0.0
        acc = []

    current_lr = optimizer.param_groups[0]['lr']

    print(f'Epoch [{epoch:03d}/{opt.epoch:03d}]'
          f' OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}')
    print(f'Current Learning Rate: {current_lr:.6f}')

    print('Per-class accuracy:')
    for i, class_acc in enumerate(acc):
        print(f'Class {i}: {class_acc:.4f}')

    if oa > best_acc:
        best_acc, best_epoch = oa, epoch  
        if (epoch >= 1):
            torch.save(optimizer.state_dict(),
                       save_path + '/weight/' + current_time + '_' + str(
                           best_acc) + '_' + opt.dataset + "_optimizer" + "Epoch" + str(epoch) + '.pth')
            torch.save(model.state_dict(),
                       save_path + '/weight/' + current_time + '_' + str(
                           best_acc) + '_' + opt.dataset + '_Net_epoch_{}.pth'.format(epoch))

            old_models = glob.glob(os.path.join(save_path, 'model', '*'))
            for old_model in old_models:
                os.remove(old_model)
            best_model_name = f'best_model_{opt.dataset}_OA{best_acc:.4f}_epoch{epoch}.pth'
            torch.save(model.state_dict(), os.path.join(save_path, 'model', best_model_name))
            logging.info(f'Saved new best model: {best_model_name}')

    print(f'Best OA: {best_acc:.4f}, Best epoch: {best_epoch:03d}')
    logging.info(f'Best OA: {best_acc:.4f}, Best epoch: {best_epoch:03d}')


if __name__ == '__main__':
    print("Start training (with cosine annealing LR)...")
    time_begin = time.time()  

    for epoch in range(opt.start_epoch, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, save_path)  
        test(test_loader, model, epoch, save_path)  

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Learning rate updated to: {current_lr:.6f}')

        time_epoch = time.time()  
        print(f"Time out:{time_epoch - time_begin:.2f}s\n")
        logging.info(f"Time out:{time_epoch - time_begin:.2f}s\n")