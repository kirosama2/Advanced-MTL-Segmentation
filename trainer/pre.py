


""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import numpy as np
from utils.metrics import eval_metrics
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from tensorboardX import SummaryWriter
from dataloader.samplers import CategoriesSampler 
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax
from utils.misc import Averager, Timer, ensure_path
from dataloader.dataset_loader import DatasetLoader as Dataset
from dataloader.mdataset_loader import mDatasetLoader as mDataset

#torch.cuda.set_device(1)
class PreTrainer(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '../logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        self.trainset = Dataset('train', self.args)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Load pre-val set
        self.valset = mDataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.labeln,self.args.num_batch , self.args.way, self.args.shot + self.args.val_query,self.args.shot)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)


        # Build pretrain model
        self.model = MtlLearner(self.args, mode='train')
        print(self.model)
        
        '''
        if self.args.pre_init_weights is not None:
            self.model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
            print(pretrained_dict.keys())
            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)   
        '''
        
        self.FL=FocalLoss()
        self.CD=CE_DiceLoss()
        self.LS=LovaszSoftmax()
        # Set optimizer 
        # Set optimizer 
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)

            # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def _reset_metrics(self):
        #self.batch_time = AverageMeter()
        #self.data_time = AverageMeter()
        #self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self,n_class):
        self.n_class=n_class
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.n_class), np.round(IoU, 3)))
        }

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['train_iou']=[]
        trlog['val_iou']=[]
        trlog['max_iou'] = 0.0
        trlog['max_iou_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'train'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_iou_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)

            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, label = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    label = batch[1]

                # Output logits for model
                logits = self.model(data)
                # Calculate train loss
                # CD loss is modified in the whole project to incorporate ony Cross Entropy loss. Modify as per requirement.
                #loss = self.FL(logits, label) + self.CD(logits,label) + self.LS(logits,label)
                loss = self.CD(logits,label) 
                
                # Calculate train accuracy
                self._reset_metrics()
                seg_metrics = eval_metrics(logits, label, self.args.num_classes)
                self._update_seg_metrics(*seg_metrics)
                pixAcc, mIoU, _ = self._get_seg_metrics(self.args.num_classes).values()

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(pixAcc)
                train_iou_averager.add(mIoU)

                # Print loss and accuracy till this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} IOU={:.4f}'.format(epoch, train_loss_averager.item(),train_acc_averager.item()*100.0,train_iou_averager.item()))
                
                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers