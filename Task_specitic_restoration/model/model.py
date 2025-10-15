import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np


logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
 
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.netG_pre = self.set_device(networks.define_G(opt))
        if opt['dist']:
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            self.netG.to(device)
            self.netG_pre.to(device)

        self.schedule_phase = None
        self.opt = opt

        # set loss and load resume state
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                [p for p in self.netG.parameters() if p.requires_grad], lr=opt['train']["optimizer"]["lr"])
            
            self.log_dict = OrderedDict()


        if self.opt['phase'] == 'test':
            # self.netG_pre = nn.DataParallel(self.netG_pre)
            # self.netG = nn.DataParallel(self.netG)
            try:
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            except Exception:
                # self.netG_pre = nn.DataParallel(self.netG_pre)
                # self.netG = nn.DataParallel(self.netG)
                self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
        else: # train
            self.netG_pre = nn.DataParallel(self.netG_pre)
            self.netG = nn.DataParallel(self.netG)
            
            self.netG_pre.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            self.netG.load_state_dict(torch.load(self.opt['path']['resume_state']), strict=True)
            # self.load_network()
            if opt['dist']:
                self.netG = DDP(self.netG, device_ids=[self.local_rank], output_device=self.local_rank,find_unused_parameters=True)

        self.loss_func = nn.MSELoss(reduction='sum').cuda()
        self.loss_func_l1 = nn.L1Loss(reduction='sum').cuda()
        self.cls_criterion = nn.BCELoss().cuda()

    def feed_data(self, data):

        dic = {}

        if self.opt['dist']:
            dic = {}
            dic['LQ'] = data['LQ'].to(self.local_rank)
            dic['GT'] = data['GT'].to(self.local_rank)
            self.data = dic
        else:
            dic['LQ'] = data['LQ']
            dic['GT'] = data['GT']


            self.data = self.set_device(dic)


    def optimize_parameters_distill(self, current_step, X, t_stride):
        
        b, c, h, w = self.data['LQ'].shape
        num_step = 10
        
        if t_stride == 10:
            t_start = 3
            # alpha = 3.60605510e-03#3.26755966e-01#1.42941651e-01#1.63855862e-01
            alpha = 0.8070461404855308#0.5711806785086991# 0.01541883 # r200l
            X_start = np.sqrt(alpha) * self.data['LQ'] + np.sqrt(1-alpha)*X[num_step-10].unsqueeze(0)

            # X_start = X[num_step-10].unsqueeze(0)
            # X_start = X[0:b]
            # print(X.shape)

                
            X_t_pre = self.netG.module.p_sample_for_distill(self.data, X_start, t_start=t_start, t_end=num_step-t_stride)
            # X_0_pre = self.netG.p_sample_for_distill(self.data['LQ'], X_t_pre, t_start=num_step-t_stride, t_end=0)
            X_0_pre = X_t_pre
            # print(X_0_pre.shape)

            # loss_T2t = self.loss_func(X_t_pre, X[t_stride].unsqueeze(0)).sum()/int(b*c*h*w) * 100
            # loss_t2O = (self.loss_func(X_0_pre, X[b*10:b*11])*self.data['LW']).sum()/int(b*c*h*w)*300
            # loss_t2G = (self.loss_func(X_0_pre, self.data['GT'])*self.data['LW']).sum()/int(b*c*h*w)*300

            loss_t2O = self.loss_func_l1(X_0_pre, X[10].unsqueeze(0)).sum()/int(b*c*h*w) * 300
            loss_t2G = self.loss_func_l1(X_0_pre, self.data['GT']).sum()/int(b*c*h*w) * 300

            t = t_start#num_step-t_stride
            l_pix = self.netG(self.data, t)
            l_pix = l_pix.sum()/int(b*c*h*w)

            loss = loss_t2O + loss_t2G + l_pix #+ loss_t2G #+ loss_G

            self.log_dict['loss'] = loss.item()
            # self.log_dict['loss_T2t'] = loss_T2t.item()
            self.log_dict['loss_t2O'] = loss_t2O.item()
            self.log_dict['loss_t2G'] = loss_t2G.item()
            # self.log_dict['loss_G'] = loss_G.item()
            self.log_dict['l_pix'] = l_pix.item()
        elif t_stride == 8:

            X_start = np.sqrt(3.60605510e-03) * self.data['LQ'] + np.sqrt(1-3.60605510e-03)*X[num_step-10].unsqueeze(0)

            # X_start = X[num_step-10].unsqueeze(0)

            X_t_pre = self.netG.p_sample_for_distill(self.data, X_start, t_start=9, t_end=num_step-t_stride)
            X_0_pre = self.netG.p_sample_for_distill(self.data, X_t_pre, t_start=num_step-t_stride, t_end=0)

            loss_T2t = self.loss_func(X_t_pre, X[t_stride].unsqueeze(0)).sum()/int(b*c*h*w) * 100
            loss_t2O = self.loss_func(X_0_pre, X[10].unsqueeze(0)).sum()/int(b*c*h*w) * 300
            loss_t2G = self.loss_func(X_0_pre, self.data['GT']).sum()/int(b*c*h*w) * 300

            t = num_step-t_stride
            l_pix = self.netG(self.data, t)
            l_pix = l_pix.sum()/int(b*c*h*w)

            loss = loss_T2t + loss_t2O + l_pix + loss_t2G #+ loss_G

            self.log_dict['loss'] = loss.item()
            self.log_dict['loss_T2t'] = loss_T2t.item()
            self.log_dict['loss_t2O'] = loss_t2O.item()
            self.log_dict['loss_t2G'] = loss_t2G.item()
            # self.log_dict['loss_G'] = loss_G.item()
            self.log_dict['l_pix'] = l_pix.item()

        elif t_stride == 6:

            X_start = np.sqrt(3.60605510e-03) * self.data['LQ'] + np.sqrt(1-3.60605510e-03)*X[num_step-10].unsqueeze(0)

            # X_start = X[num_step-10].unsqueeze(0)

            X_t_pre = self.netG.p_sample_for_distill(self.data, X_start, t_start=9, t_end=num_step-t_stride)
            X_t_1_pre = self.netG.p_sample_for_distill(self.data, X_t_pre, t_start=num_step-t_stride, t_end=num_step-t_stride-1)
            X_t_2_pre = self.netG.p_sample_for_distill(self.data, X_t_1_pre, t_start=num_step-t_stride-1, t_end=num_step-t_stride-2)
            X_0_pre = self.netG.p_sample_for_distill(self.data, X_t_2_pre, t_start=num_step-t_stride-2, t_end=0)

            loss_T2t = (self.loss_func(X_t_pre, X[t_stride].unsqueeze(0)).sum()/int(b*c*h*w) + \
                        self.loss_func(X_t_1_pre, X[t_stride+1].unsqueeze(0)).sum()/int(b*c*h*w) + \
                        self.loss_func(X_t_2_pre, X[t_stride+2].unsqueeze(0)).sum()/int(b*c*h*w)) * 100
            loss_t2O = self.loss_func(X_0_pre, X[10].unsqueeze(0)).sum()/int(b*c*h*w) * 300
            loss_t2G = self.loss_func(X_0_pre, self.data['GT']).sum()/int(b*c*h*w) * 600

            t = num_step-t_stride
            l_pix = self.netG(self.data, t)
            l_pix = l_pix.sum()/int(b*c*h*w)

            loss = loss_T2t + loss_t2O + l_pix + loss_t2G #+ loss_G

            self.log_dict['loss'] = loss.item()
            self.log_dict['loss_T2t'] = loss_T2t.item()
            self.log_dict['loss_t2O'] = loss_t2O.item()
            self.log_dict['loss_t2G'] = loss_t2G.item()
            # self.log_dict['loss_G'] = loss_G.item()
            self.log_dict['l_pix'] = l_pix.item()

        
        
        loss.backward()
        if current_step % 2 == 0:
            # loss_g.backward()
            self.optG.step() 
            self.optG.zero_grad()

    def optimize_parameters_reinforce(self, current_step, t, X_sde, update_Gpre):
        
        b, c, h, w = self.data['LQ'].shape

        X_ode = self.netG.p_sample_loop_for_ode(self.data['LQ'], t, continous=False)
        # X_ode = self.netG.module.p_sample_loop_for_ode(self.data['LQ'], t, continous=False)
        
        loss_t = self.loss_func(X_ode[t].unsqueeze(0), X_sde[t].unsqueeze(0)).sum()/int(b*c*h*w) 
        loss_GT = self.loss_func(X_ode[-1].unsqueeze(0), self.data['GT']).sum()/int(b*c*h*w)

        l_pix = self.netG(self.data, t)
        l_pix = l_pix.sum()/int(b*c*h*w)

        loss = l_pix + loss_t + loss_GT
        loss.backward()
        if current_step % 2 == 0:
            # loss_g.backward()
            self.optG.step() 
            self.optG.zero_grad()

        # set log
        self.log_dict['loss'] = loss.item()
        self.log_dict['loss_t'] = loss_t.item()
        self.log_dict['loss_GT'] = loss_GT.item()        
        self.log_dict['l_pix'] = l_pix.item()


    def optimize_parameters(self):

        self.optG.zero_grad()
        # print(self.data['GT'].device, self.netG.device)
        l_pix = self.netG(self.data)

        b, c, h, w = self.data['LQ'].shape

        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def test_cost(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                cost_10_to_t_all, cost_t_to_0_all = self.netG.module.super_resolution_for_cost(
                    self.data['LQ'], self.data['GT'], continous)
                
            else:
                if self.opt['dist']:
                    cost_10_to_t_all, cost_t_to_0_all = self.netG.module.super_resolution_for_cost(self.data['LQ'], self.data['GT'], continous)
                else:
                    cost_10_to_t_all, cost_t_to_0_all = self.netG.super_resolution_for_cost(self.data['LQ'], self.data['GT'], continous)
            return cost_10_to_t_all, cost_t_to_0_all


    def test_distill(self, NFE, dataset, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution_distill(
                    self.data, continous)
            else:
                if self.opt['dist']:
                    self.SR = self.netG.module.super_resolution_distill(self.data['LQ'], NFE, dataset, continous)
                else:
                    self.SR = self.netG.super_resolution_distill(self.data, NFE, dataset, continous)
        self.netG.train()


    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['LQ'], continous)
                
            else:
                if self.opt['dist']:
                    self.SR = self.netG.module.super_resolution(self.data['LQ'], continous)
                else:
                    self.SR = self.netG.super_resolution(self.data['LQ'], continous)
        self.netG.train()

    def sample_get_p(self, X_t_1_pre, t, continous=False):
        # self.netG.eval()
        # with torch.no_grad():
        if isinstance(self.netG, nn.DataParallel):
            pe_t_1 = self.netG.module.p_sample_loop_for_calcu_pe(self.data['LQ'], X_t_1_pre, t, continous)
            
        else:
            if self.opt['dist']:
                pe_t_1 = self.netG.module.p_sample_loop_for_calcu_pe(self.data['LQ'], X_t_1_pre, t, continous)
            else:
                pe_t_1 = self.netG.p_sample_loop_for_calcu_pe(self.data['LQ'], X_t_1_pre, t, continous)
        # self.netG.train()
        return pe_t_1

    def sample_train_dis(self, continous=False):
        self.netG_pre.eval()
        with torch.no_grad():
            if isinstance(self.netG_pre, nn.DataParallel):
                self.SR, self.fx_t, self.p_t_1 = self.netG_pre.module.super_resolution_for_reward(
                    self.data['LQ'], continous)
                
            else:
                if self.opt['dist']:
                    self.SR, self.fx_t, self.p_t_1 = self.netG_pre.module.super_resolution_for_reward(self.data['LQ'], continous)
                else:
                    self.SR, self.fx_t, self.p_t_1 = self.netG_pre.super_resolution_for_reward(self.data['LQ'], continous)

    def get_current_visuals_reward(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['HQ'] = self.SR
            out_dict['F_xt'] = self.fx_t
            out_dict['P_t_1'] = self.p_t_1
            out_dict['INF'] = self.data['LQ'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
            else:
                out_dict['LQ'] = out_dict['INF']
        return out_dict


    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

        # if not self.opt['uncertainty_train']:
        #     if isinstance(self.netGU, nn.DataParallel):
        #         self.netGU.module.set_loss(self.device)
        #     else:
        #         self.netGU.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):

        if self.opt['dist']:
            # local_rank = torch.distributed.get_rank()
            device = torch.device("cuda", self.local_rank)
            if self.schedule_phase is None or self.schedule_phase != schedule_phase:
                self.schedule_phase = schedule_phase
                if isinstance(self.netG, nn.DataParallel):
                    self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
                    self.netG_pre.module.set_new_noise_schedule(schedule_opt, self.device)
                else:
                    self.netG.set_new_noise_schedule(schedule_opt, device)
                    self.netG_pre.set_new_noise_schedule(schedule_opt, device)

                # if not self.opt['uncertainty_train']:
                #     if isinstance(self.netGU, nn.DataParallel):
                #         self.netGU.module.set_new_noise_schedule(
                #             schedule_opt, self.device)
                #     else:
                #         self.netGU.set_new_noise_schedule(schedule_opt, device)
        else:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
                self.netG_pre.module.set_new_noise_schedule(schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
                self.netG_pre.set_new_noise_schedule(schedule_opt, self.device)

            # if not self.opt['uncertainty_train']:
            #     if isinstance(self.netGU, nn.DataParallel):
            #         self.netGU.module.set_new_noise_schedule(
            #             schedule_opt, self.device)
            #     else:
            #         self.netGU.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['HQ'] = self.SR.detach().float().cpu()
            # out_dict['UT'] = self.utmap.detach().float().cpu()
            # out_dict['Ill'] = self.data['Ill'].detach().float().cpu()
            out_dict['INF'] = self.data['LQ'].detach().float().cpu()
            out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
            else:
                out_dict['LQ'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            # self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
            self.opt['path']['checkpoint'], 'best_gen.pth')
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # # opt
        # opt_state = {'epoch': epoch, 'iter': iter_step,
        #              'scheduler': None, 'optimizer': None}
        # opt_state['optimizer'] = self.optG.state_dict()
        # torch.save(opt_state, opt_path)

        # if self.opt['uncertainty_train']:
        #     ut_gen_path = os.path.join(
        #         './checkpoints/uncertainty/', 'latest_gen.pth'.format(iter_step, epoch))
        #     ut_opt_path = os.path.join(
        #         './checkpoints/uncertainty/', 'latest_opt.pth'.format(iter_step, epoch))
        #     torch.save(state_dict, ut_gen_path)
        #     torch.save(opt_state, ut_opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            # opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            # network = nn.DataParallel(network).cuda()

            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                # opt = torch.load(opt_path)
                # self.optG.load_state_dict(opt['optimizer'])
                # self.begin_step = opt['iter']
                # self.begin_epoch = opt['epoch']
                self.begin_step = 0
                self.begin_epoch = 0
