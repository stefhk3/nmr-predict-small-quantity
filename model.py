import torch
from torch_geometric.data import Batch,Data

NO_GRAPH_FEATURES=128

ENCODING_NODE=64
ENCODING_EDGE=32

HIDDEN_NODE=128
HIDDEN_EDGE=64
HIDDEN_GRAPH=128

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(LazyLinear(HIDDEN_EDGE), LeakyReLU(),LazyBatchNorm1d(),
                            LazyLinear(HIDDEN_EDGE), LeakyReLU(),LazyBatchNorm1d(),
                            LazyLinear(ENCODING_EDGE)).apply(init_weights)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(LazyLinear(HIDDEN_NODE), LeakyReLU(), LazyBatchNorm1d(),
                              LazyLinear(HIDDEN_NODE), LeakyReLU(), LazyBatchNorm1d(), #torch.nn.Dropout(0.17),
                              LazyLinear(HIDDEN_NODE)).apply(init_weights)
        self.node_mlp_2 = Seq(LazyLinear(HIDDEN_NODE), LeakyReLU(),LazyBatchNorm1d(), #torch.nn.Dropout(0.17),
                              LazyLinear(HIDDEN_NODE), LeakyReLU(),LazyBatchNorm1d(),
                              LazyLinear(ENCODING_NODE)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(LazyLinear(HIDDEN_GRAPH), LeakyReLU(),LazyBatchNorm1d(), #torch.nn.Dropout(0.17),
                              LazyLinear(HIDDEN_GRAPH), LeakyReLU(),LazyBatchNorm1d(),
                              LazyLinear(NO_GRAPH_FEATURES)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        print(x, edge_index, edge_attr, u, batch)
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        node_aggregate = scatter_add(x, batch, dim=0)
        edge_aggregate = scatter_add(edge_attr, batch[col], dim=0)
        out = torch.cat([node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out)

class GNN_FULL_CLASS(torch.nn.Module):
    def __init__(self, NO_MP):
        super(GNN_FULL_CLASS,self).__init__()
        #Meta Layer for Message Passing
        self.meta = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        
        #Edge Encoding MLP
        self.encoding_edge=Seq(LazyLinear(ENCODING_EDGE), LeakyReLU(), LazyBatchNorm1d(),
                               LazyLinear(ENCODING_EDGE), LeakyReLU(), LazyBatchNorm1d(),
                               LazyLinear(ENCODING_EDGE)).apply(init_weights)

        self.encoding_node = Seq(LazyLinear(ENCODING_NODE), LeakyReLU(),LazyBatchNorm1d(),
                                 LazyLinear(ENCODING_NODE), LeakyReLU(),LazyBatchNorm1d(),
                                 LazyLinear(ENCODING_NODE)).apply(init_weights)

        self.mlp_last = Seq(LazyLinear(HIDDEN_NODE), LeakyReLU(),#torch.nn.Dropout(0.10),
                            LazyBatchNorm1d(),
                            LazyLinear(HIDDEN_NODE), LeakyReLU(),
                            LazyBatchNorm1d(),
                            LazyLinear(1)).apply(init_weights)
        
        self.no_mp = NO_MP
        

    def forward(self,dat):
        #Extract the data from the batch
        x, ei, ea, u, btc = dat.x, dat.edge_index, dat.edge_attr, dat.y, dat.batch

        # Embed the node and edge features
        enc_x = self.encoding_node(x)
        enc_ea = self.encoding_edge(ea)
    
        #Create the empty molecular graphs for feature extraction, graph level one
        u=torch.full(size=(x.size()[0], 1), fill_value=0.1, dtype=torch.float)

        #Message-Passing
        for _ in range(self.no_mp):
            enc_x, enc_ea, u = self.meta(x = enc_x, edge_index = ei, edge_attr = enc_ea, u = u, batch = btc)
        
        targs = self.mlp_last(enc_x)

        return targs
    
def init_model(NO_MP, lr, wd):
  # Model
  NO_MP = NO_MP
  model = GNN_FULL_CLASS(NO_MP)

  # Optimizer
  LEARNING_RATE = lr
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)

  # Criterion
  #criterion = torch.nn.MSELoss()
  criterion = torch.nn.L1Loss()
  return model, optimizer, criterion

def train(model, criterion, optimizer, loader):
    loss_sum = 0
    for batch in loader:
        # Forward pass and gradient descent
        labels = batch.y
        predictions = model(batch)
        loss = criterion(predictions[torch.isfinite(labels)], labels[torch.isfinite(labels)])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    return loss_sum/len(loader)



def evaluate(model, criterion, loader):
    loss_sum = 0
    with torch.no_grad():
        for batch in loader:
            # Forward pass
            labels = batch.y
            predictions = model(batch)
            loss = criterion(predictions[torch.isfinite(labels)], labels[torch.isfinite(labels)])
            loss_sum += loss.item()
    return loss_sum/len(loader)

def chunk_into_n(lst, n):
  size = math.ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )

def get_data_loaders(data, split, batch_size):
    random.Random().shuffle(data)
    training_data = all_data[:int(len(all_data)*split)]
    testing_data = all_data[int(len(all_data)*split):]
    train_loader = DataLoader(training_data, batch_size = batch_size)
    test_loader = DataLoader(testing_data, batch_size = batch_size)
    return train_loader, test_loader

def getBondElements(bond):
  
    a= bond.GetEndAtom().GetSymbol()
    b = bond.GetBeginAtom().GetSymbol()
    return a+b if a<b else b+a


def train_model(train_set, test_set, split, batch_size, epochs, weight_decay=0.005, learning_rate=0.0003, NO_MP=7):
    NO_MP = NO_MP
    model = GNN_FULL_CLASS(NO_MP)

    # Optimizer
    LEARNING_RATE = learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

    # Criterion
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    model.train()

    train_loader = DataLoader(train_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)

    for epoch in range(epochs):
        tloss = train(model, criterion, optimizer, train_loader)
        train_err = evaluate(model, criterion, train_loader)
        test_err = evaluate(model, criterion, test_loader)
    #print(train_err, test_err)
    return (test_err, tloss, train_err)
