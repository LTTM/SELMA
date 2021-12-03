import torch

from utils.cnames import names_dict
from utils.idsmask import ids_dict

class Metrics:
    def __init__(self, class_set, log_colors=False, device='cuda'):
        self.class_set = class_set
        self.name_classes = names_dict[class_set]
        self.num_classes = len(self.name_classes)
        self.log_colors = log_colors
        self.device = device
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=device)
        
        self.color_dict = {'cyan'  : '\033[96m',
                           'green' : '\033[92m',
                           'yellow': '\033[93m',
                           'red'   : '\033[91m'}

    def __genterate_cm__(self, pred, gt):                                                       #       preds                    
        mask = (gt >= 0) & (gt < self.num_classes)                                              #     +------- 
        combinations = self.num_classes*gt[mask] + pred[mask] # 0 <= comb <= num_classes^2-1    #   l | . . . 
        cm_entries = torch.bincount(combinations, minlength=self.num_classes**2)                #   b | . . .
        return cm_entries.reshape(self.num_classes, self.num_classes)                           #   s | . . . 
        
    def add_sample(self, pred, gt):
        assert pred.shape == gt.shape, "Prediction and Ground Truth must have the same shape"
        self.confusion_matrix += self.__genterate_cm__(pred, gt) # labels along rows, predictions along columns
        
    def PA(self):
        # Pixel Accuracy (Recall) = TP/(TP+FN)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=1)
        
    def PP(self):
        # Pixel Precision = TP/(TP+FP)
        return torch.diagonal(self.confusion_matrix)/self.confusion_matrix.sum(dim=0)
        
    def IoU(self):
        # Intersection over Union = TP/(TP+FP+FN)
        return torch.diagonal(self.confusion_matrix)/(self.confusion_matrix.sum(dim=1)+self.confusion_matrix.sum(dim=0)-torch.diagonal(self.confusion_matrix))

    def percent_mIoU(self, target_class_set=None):
        keep_ids = ids_dict[self.class_set][target_class_set] if target_class_set is not None else None
        return 100*self.nanmean(self.IoU(), keep_ids)

    @staticmethod
    def nanmean(tensor, keep_ids=None):
        m = torch.isnan(tensor[keep_ids]) if keep_ids is not None else torch.isnan(tensor)
        return torch.mean(tensor[keep_ids][~m])if keep_ids is not None else torch.mean(tensor[~m])
        
    @staticmethod
    def nanstd(tensor, keep_ids=None):
        m = torch.isnan(tensor[keep_ids]) if keep_ids is not None else torch.isnan(tensor)
        return torch.std(tensor[keep_ids][~m])if keep_ids is not None else torch.std(tensor[~m])
        
    def color_tuple(self, val, c):
        return (self.color_dict[c], val, '\033[0m')
    
    @staticmethod
    def get_color(val, mean, std):
        if val < mean-std:
            return 'red'
        if val < mean:
            return 'yellow'
        if val < mean+std:
            return 'green'
        return 'cyan'
        
    def str_class_set(self, target_class_set=None):
        return self.__str__(target_class_set)
    
    def __str__(self, target_class_set=None):
        out = "="*39+'\n'
        out += "  Class\t\t PA %\t PP %\t IoU%\n"
        out += "-"*39+'\n'
        
        keep_ids = ids_dict[self.class_set][target_class_set] if target_class_set is not None else None
            
        
        pa, pp, iou = 100*self.PA(), 100*self.PP(), 100*self.IoU()
        mpa, mpp, miou = self.nanmean(pa, keep_ids), self.nanmean(pp, keep_ids), self.nanmean(iou, keep_ids)
        spa, spp, siou = self.nanstd(pa, keep_ids), self.nanstd(pp, keep_ids), self.nanstd(iou, keep_ids)
        for i, n in enumerate(self.name_classes):
            if target_class_set is not None:
                if i not in keep_ids:
                    continue
            npa, npp, niou = pa[i], pp[i], iou[i]
            pad = "" if len(n) >= 6 else "\t"
            if self.log_colors:
                cpa, cpp, ciou = self.get_color(npa, mpa, spa), self.get_color(npp, mpp, spp), self.get_color(niou, miou, siou)
                tpa, tpp, tiou = self.color_tuple(npa, cpa), self.color_tuple(npp, cpp), self.color_tuple(niou, ciou)
                out += "  %s\t%s %s%.1f%s\t %s%.1f%s\t %s%.1f%s\n"%(n, pad, *tpa, *tpp, *tiou)
            else:
                out += "  %s\t%s %.1f\t %.1f\t %.1f\n"%(n, pad, npa, npp, niou)
        out += "-"*39+'\n'
        out += "  Average\t %.1f\t %.1f\t %.1f\n"%(mpa, mpp, miou)
        out += "  Std. Dev.\t %.1f\t %.1f\t %.1f\n"%(spa, spp, siou)
        out += "="*39+'\n'
        return out
            
# if len(n)>=6:
    # out += "  %s\t %.1f\t %.1f\t %.1f\n"%(n, npa, npp, niou)
# else:
    # out += "  %s\t\t %.1f\t %.1f\t %.1f\n"%(n, npa, npp, niou)
