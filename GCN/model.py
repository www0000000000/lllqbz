import torch
from torch import nn
from torch_geometric.nn import RGCNConv,GCNConv,GATConv
import torch.nn.functional as F

class DRGCN(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class DRGCN1(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN1, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        ''''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=d
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class DRGCN2(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN2, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        '''
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=t
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class DRGCN3(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN3, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=n
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
            
class DRGCN4(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN4, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        '''
        c=self.linear_relu_cat_prop(cat_prop)
        x=c
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class DRGCN12(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN12, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        '''
        x=torch.cat((d,t),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class DRGCN34(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN34, self).__init__()
        self.dropout = dropout
        '''
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )

        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        '''
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/2)),
            nn.LeakyReLU()
        )
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        '''
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        '''
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class DGCN(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gcn1=GCNConv(embedding_dimension,embedding_dimension)
        self.gcn2=GCNConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gcn1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gcn2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    
class BotGAT(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(BotGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.gat1=GATConv(embedding_dimension,int(embedding_dimension/4),heads=4)
        self.gat2=GATConv(embedding_dimension,embedding_dimension)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.gat1(x,edge_index)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.gat2(x,edge_index)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
            
class DRGCN_4layers(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN_4layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x
    

class DRGCN_8layers(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=6,cat_prop_size=11,embedding_dimension=128,dropout=0.3):
        super(DRGCN_8layers, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2=nn.Linear(embedding_dimension,2)
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear_relu_output1(x)
        x=self.linear_output2(x)
            
        return x