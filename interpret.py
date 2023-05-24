# Semi-Supervised Model Interpretation

# Import Packages (fastai for optimizing inference)
import os
from fastai.text import *
from utils.quick_load import *
from utils.model import *
from sklearn.metrics import *
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_curve, auc


# Confusion Matrix Operation:

def confusion_matrix(self, slice_size=1):
    x=torch.arange(0,self.data.c)
    if slice_size is None: cm = ((self.pred_class==x[:,None]) & (self.y_true==x[:,None,None])).sum(2)
    else:
        cm = torch.zeros(self.data.c, self.data.c, dtype=x.dtype)
        for i in range(0, self.y_true.shape[0], slice_size):
            cm_slice = ((self.pred_class[i:i+slice_size]==x[:,None]) & (self.y_true[i:i+slice_size]==x[:,None,None])).sum(2)
            torch.add(cm, cm_slice, out=cm)
    return to_np(cm)
# Attention Operation:

def _eval_dropouts(mod):
        module_name =  mod.__class__.__name__
        if 'Dropout' in module_name or 'BatchNorm' in module_name: mod.training = False
        for module in mod.children(): _eval_dropouts(module)


def intrinsic_perturb(self, text, class_id=None):
    self.model.train()
    _eval_dropouts(self.model)
    self.model.zero_grad()
    self.model.reset()
    ids = self.data.one_item(text)[0]
    emb = self.model[0].module.encoder(ids).detach().requires_grad_(True)
    lstm_output = self.model[0].module(emb, from_embeddings=True)
    self.model.eval()
    cl = self.model[1](lstm_output + (torch.zeros_like(ids).byte(),))[0].softmax(dim=-1)
    if class_id is None: 
        class_id = cl.argmax()
    cl[0][class_id].backward()
    attn = emb.grad.squeeze().abs().sum(dim=-1)
    attn /= attn.max()
    tokens = self.data.single_ds.reconstruct(ids[0])
    return tokens, attn


intrinsic_perturb(interpretLearn, 'Stable postsurgical appearance of graft replacement of the ascending aorta, arch, and descending thoracic aorta. Minimal decrease in size of fluid collection surrounding the descending thoracic graft and adjacent small pleural effusion.')

# Generate Document Encodings:

def encoding_generator(self, text:str, n_words:int=1, no_unk:bool=True, temperature:float=1., min_p:float=None, sep:str=' ',
            decoder=decode_spec_tokens):
    ds = self.data.single_dl.dataset
    xb,yb = self.data.one_item(text)
    new_idx = []
    x = 0
    x = pred_batch(self,batch=(xb,yb))
    return x


def _loss_func_name2activ(name:str, axis:int=-1):
    res = loss_func_name2activ[name]
    if res == F.softmax: res = partial(F.softmax, dim=axis)
    return res

loss_func_name2activ = {'cross_entropy_loss': F.softmax, 'nll_loss': torch.exp, 'poisson_nll_loss': torch.exp,
    'kl_div_loss': torch.exp, 'bce_with_logits_loss': torch.sigmoid, 'cross_entropy': F.softmax,
    'kl_div': torch.exp, 'binary_cross_entropy_with_logits': torch.sigmoid,
}


def _loss_func2activ(loss_func):
    if getattr(loss_func,'keywords',None):
        if not loss_func.keywords.get('log_input', True): return
    axis = getattr(loss_func, 'axis', -1)
    # flattened loss
    loss_func = getattr(loss_func, 'func', loss_func)
    # could have a partial inside flattened loss! Duplicate on purpose.
    loss_func = getattr(loss_func, 'func', loss_func)
    cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name == 'mix_up_loss':
        loss_func = loss_func.crit
        cls_name = camel2snake(loss_func.__class__.__name__)
    if cls_name in loss_func_name2activ:
        if cls_name == 'poisson_nll_loss' and (not getattr(loss_func, 'log_input', True)): return
        return _loss_func_name2activ(cls_name, axis)
    if getattr(loss_func,'__name__','') in loss_func_name2activ:
        return _loss_func_name2activ(loss_func.__name__, axis)
    return noop


def loss_batch(model:nn.Module, xb:Tensor, yb:Tensor, loss_func:OptLossFunc=None, opt:OptOptimizer=None,
               cb_handler:Optional[CallbackHandler]=None)->Tuple[Union[Tensor,int,float,str]]:
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    cb_handler = ifnone(cb_handler, CallbackHandler())
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    
    out = model(*xb)
    return out[1][-1][-1]


def pred_batch(self, ds_type:DatasetType=DatasetType.Valid, batch:Tuple=None, reconstruct:bool=False, with_dropout:bool=False) -> List[Tensor]:
        if batch is not None: xb,yb = batch
        else: xb,yb = self.data.one_batch(ds_type, detach=False, denorm=False)
        cb_handler = CallbackHandler(self.callbacks)
        xb,yb = cb_handler.on_batch_begin(xb,yb, train=False)
        with torch.no_grad():
            if not with_dropout: preds = loss_batch(self.model.eval(), xb, yb, cb_handler=cb_handler)
            else: preds = loss_batch(self.model.eval().apply(self.apply_dropout), xb, yb, cb_handler=cb_handler)
            res = _loss_func2activ(self.loss_func)(preds[0])
        if not reconstruct: return res
        res = res.detach().cpu()
        ds = self.dl(ds_type).dataset
        norm = getattr(self.data, 'norm', False)
        if norm and norm.keywords.get('do_y',False):
            res = self.data.denorm(res, do_x=True)
        return [ds.reconstruct(o) for o in res]

    def opt_th(preds, targs, start=0.01, end=1, step=0.01):
        ths = np.arange(start,end,step)
        thresholds=[]
        thres = 0
        best_score = 0
        for th in ths:
            thresholds.append(fbeta_score(targs, (preds>th), 2, average='binary'))
            if fbeta_score(targs, (preds>th), 2, average='binary') > best_score:
                best_score = fbeta_score(targs, (preds>th), 2, average='binary')
        idx = np.argmax(thresholds)
        print('Best threshold = ', ths[idx])
        print('Best F-Score = ', best_score)

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    text_data = load_data("../data/", 'reports_all.pkl')
    labeled_data = load_data('../data','AAA.pkl')

    # Load in saved final model
    # - `load_ccds_model` is a function in the architecture folder that automatically loads in the ccds model, but must be customized depending on the parameter changes made during training
    learn = load_ccds_model(text_data=text_data,
                    labeled_data=labeled_data,
                    path_to_lm='language_modelv1',
                    path_to_classifier='AAA',
                    decoder_layer_sizes=[50],
                    decoder_dropout=[0.1])


    # Interpretation model: from PyTorch --> FastAI
    interpretLearn = TextClassificationInterpretation.from_learner(learn)
    
    cm = confusion_matrix(interpretLearn)

    encoding = encoding_generator(learn,'diseases present is this file hemorrhage')

    print(len(encoding))

    # get test results
    y_score,y_test=learn.get_preds(DatasetType.Test)
    y_test = y_test.numpy()
    y_score = y_score.numpy()
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1], pos_label=1,)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.30f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Classification Impressions')
    plt.legend(loc="lower right")
    plt.show()
    opt_th(y_score[:,1], y_test)