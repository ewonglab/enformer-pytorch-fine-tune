import argparse
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from finetune.fine_tune_tidy import EnformerFineTuneModel
from data_sets.mouse_8_25 import mouse_8_25


# allantois
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.7307692170143127     │
# │       ptl/test_aupr       │    0.8426597118377686     │
# │      ptl/test_auroc       │     0.827689528465271     │
# │     ptl/test_f1_score     │    0.7717121839523315     │
# │       ptl/test_loss       │    0.5905972719192505     │
# │       ptl/test_mcc        │    0.48716554045677185    │
# │    ptl/test_precision     │    0.6761611104011536     │
# │      ptl/test_recall      │    0.8987179398536682     │
# └───────────────────────────┴───────────────────────────┘

# cardiom
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.8045976758003235     │
# │       ptl/test_aupr       │    0.8895797729492188     │
# │      ptl/test_auroc       │    0.8885513544082642     │
# │     ptl/test_f1_score     │    0.7999722957611084     │
# │       ptl/test_loss       │    0.4674491286277771     │
# │       ptl/test_mcc        │    0.6103220582008362     │
# │    ptl/test_precision     │    0.8242945671081543     │
# │      ptl/test_recall      │    0.7771029472351074     │
# └───────────────────────────┴───────────────────────────┘

    
# NMP 
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.6268292665481567     │
# │       ptl/test_aupr       │    0.8560953140258789     │
# │      ptl/test_auroc       │    0.8586573004722595     │
# │     ptl/test_f1_score     │    0.7339987754821777     │
# │       ptl/test_loss       │    1.9869482517242432     │
# │       ptl/test_mcc        │    0.3558703362941742     │
# │    ptl/test_precision     │    0.5814154744148254     │
# │      ptl/test_recall      │    0.9952830076217651     │
# └───────────────────────────┴───────────────────────────┘


# erythroid
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.9154229164123535     │
# │       ptl/test_aupr       │    0.9721502065658569     │
# │      ptl/test_auroc       │    0.9742027521133423     │
# │     ptl/test_f1_score     │    0.9173167943954468     │
# │       ptl/test_loss       │    0.21146658062934875    │
# │       ptl/test_mcc        │    0.8313151001930237     │
# │    ptl/test_precision     │    0.9087165594100952     │
# │      ptl/test_recall      │    0.9264705777168274     │
# └───────────────────────────┴───────────────────────────┘


# exe_endo
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.9166666269302368     │
# │       ptl/test_aupr       │    0.9617514610290527     │
# │      ptl/test_auroc       │    0.9682539701461792     │
# │     ptl/test_f1_score     │    0.8999999761581421     │
# │       ptl/test_loss       │    0.25329846143722534    │
# │       ptl/test_mcc        │    0.8285714387893677     │
# │    ptl/test_precision     │    0.8999999761581421     │
# │      ptl/test_recall      │    0.8999999761581421     │
# └───────────────────────────┴───────────────────────────┘

# forebrain
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.7938144207000732     │
# │       ptl/test_aupr       │    0.8819767236709595     │
# │      ptl/test_auroc       │    0.8792734742164612     │
# │     ptl/test_f1_score     │    0.8276476860046387     │
# │       ptl/test_loss       │    0.5351853370666504     │
# │       ptl/test_mcc        │    0.5973876714706421     │
# │    ptl/test_precision     │    0.7501831650733948     │
# │      ptl/test_recall      │    0.9230769276618958     │
# └───────────────────────────┴───────────────────────────┘

# gut checkpoint0008
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │         0.7109375         │
# │       ptl/test_aupr       │    0.9450875520706177     │
# │      ptl/test_auroc       │    0.9379333257675171     │
# │     ptl/test_f1_score     │    0.7848837375640869     │
# │       ptl/test_loss       │    0.7023569345474243     │
# │       ptl/test_mcc        │    0.5009557008743286     │
# │    ptl/test_precision     │    0.6459707021713257     │
# │      ptl/test_recall      │            1.0            │
# └───────────────────────────┴───────────────────────────┘

# mesenchyme checkpoint0002
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.8301886916160583     │
# │       ptl/test_aupr       │    0.9052236080169678     │
# │      ptl/test_auroc       │     0.905646562576294     │
# │     ptl/test_f1_score     │    0.8434782028198242     │
# │       ptl/test_loss       │     0.474099338054657     │
# │       ptl/test_mcc        │    0.6732609272003174     │
# │    ptl/test_precision     │     0.77585768699646      │
# │      ptl/test_recall      │    0.9241654872894287     │
# └───────────────────────────┴───────────────────────────┘

# mid_hindbrain checkpoint0006
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │     0.790123462677002     │
# │       ptl/test_aupr       │    0.8882557153701782     │
# │      ptl/test_auroc       │    0.8877661228179932     │
# │     ptl/test_f1_score     │    0.8218600749969482     │
# │       ptl/test_loss       │     0.501795768737793     │
# │       ptl/test_mcc        │    0.5916914939880371     │
# │    ptl/test_precision     │    0.7440087199211121     │
# │      ptl/test_recall      │    0.9180509448051453     │
# └───────────────────────────┴───────────────────────────┘

# mixed_mesoderm checkpoint_000008
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.7241379022598267     │
# │       ptl/test_aupr       │     0.855053186416626     │
# │      ptl/test_auroc       │    0.8404761552810669     │
# │     ptl/test_f1_score     │    0.6794872283935547     │
# │       ptl/test_loss       │    0.48412126302719116    │
# │       ptl/test_mcc        │    0.4656839966773987     │
# │    ptl/test_precision     │    0.8090909123420715     │
# │      ptl/test_recall      │    0.5857143402099609     │
# └───────────────────────────┴───────────────────────────┘

# neuralcrest checkpoint_000006
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │        0.81640625         │
# │       ptl/test_aupr       │    0.9320405721664429     │
# │      ptl/test_auroc       │    0.9260401725769043     │
# │     ptl/test_f1_score     │    0.8457899689674377     │
# │       ptl/test_loss       │    0.5146384239196777     │
# │       ptl/test_mcc        │    0.6602393984794617     │
# │    ptl/test_precision     │    0.7498985528945923     │
# │      ptl/test_recall      │    0.9698100090026855     │
# └───────────────────────────┴───────────────────────────┘

# paraxial_meso checkpoint_000006
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.7582781314849854     │
# │       ptl/test_aupr       │    0.8515762090682983     │
# │      ptl/test_auroc       │    0.8462328910827637     │
# │     ptl/test_f1_score     │    0.7921293377876282     │
# │       ptl/test_loss       │    0.5109297037124634     │
# │       ptl/test_mcc        │    0.5226922631263733     │
# │    ptl/test_precision     │    0.7243812084197998     │
# │      ptl/test_recall      │    0.8742088675498962     │
# └───────────────────────────┴───────────────────────────┘ 

# pharyngeal_meso checkpoint_000006
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.6911764740943909     │
# │       ptl/test_aupr       │    0.8046243190765381     │
# │      ptl/test_auroc       │    0.8140138387680054     │
# │     ptl/test_f1_score     │           0.75            │
# │       ptl/test_loss       │    0.5667592883110046     │
# │       ptl/test_mcc        │    0.4382610023021698     │
# │    ptl/test_precision     │    0.6316425204277039     │
# │      ptl/test_recall      │    0.9264705777168274     │
# └───────────────────────────┴───────────────────────────┘

# spinalcord checkpoint_000009
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │     0.764976978302002     │
# │       ptl/test_aupr       │    0.8542972803115845     │
# │      ptl/test_auroc       │    0.8546711206436157     │
# │     ptl/test_f1_score     │    0.7968127727508545     │
# │       ptl/test_loss       │    0.4883466064929962     │
# │       ptl/test_mcc        │    0.5450083017349243     │
# │    ptl/test_precision     │    0.7168551087379456     │
# │      ptl/test_recall      │    0.8968790173530579     │
# └───────────────────────────┴───────────────────────────┘

# surface_ecto checkpoint_000009
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │     ptl/test_accuracy     │    0.6639344692230225     │
# │       ptl/test_aupr       │    0.9404280185699463     │
# │      ptl/test_auroc       │    0.9362602829933167     │
# │     ptl/test_f1_score     │     0.751663088798523     │
# │       ptl/test_loss       │    0.6563304662704468     │
# │       ptl/test_mcc        │    0.4247581958770752     │
# │    ptl/test_precision     │    0.6052505970001221     │
# │      ptl/test_recall      │    0.9919354915618896     │
# └───────────────────────────┴───────────────────────────┘


if __name__ == '__main__':
    # pytorch lightning to test model
    # test_model()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", type=str, required=True, help="Type of the cell")
    parser.add_argument("--lr", type=float, required=True, help="Type of the cell")
    parser.add_argument("--layer_size", type=int, required=True, help="Type of the cell")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Type of the cell")

    # Parse the arguments
    args = parser.parse_args()

    # This is the cell type
    cell_type = args.cell_type
    lr = args.lr
    layer_size = args.layer_size
    checkpoint_dir = args.checkpoint_dir

    # cell_type = 'surface_ecto'


    config = {
        'lr' : lr,
        'layer_size' : layer_size,
    }
    
    model = EnformerFineTuneModel.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint.ckpt"), pretrained_model_name='EleutherAI/enformer-official-rough', config=config)
    trainer_2 = pl.Trainer(accelerator="gpu", devices="auto", deterministic=True)
    test_dataloader_2 = DataLoader(mouse_8_25(cell_type=cell_type, data_class='test'), batch_size=8)
    trainer_2.test(model, dataloaders=[test_dataloader_2])
