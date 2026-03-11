import torch
import torch.nn as nn
from tqdm import tqdm
from .model import TopoSAMFlow
from .evaluate import evaluate

def train_epoch(model, dataloader, optimizer, device, epoch, 
                sam_model=None, log_interval=10):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, targets, boxes) in enumerate(progress_bar):
        images = images.to(device)
        boxes = boxes.to(device) if boxes is not None else None
        
        optimizer.zero_grad()
        
        # Forward pass
        phi, aux = model(images, boxes=boxes, sam_model=sam_model)
        
        # Compute loss
        loss, loss_dict = model.get_loss(phi, aux, images, 
                                        ground_truth=targets, boxes=boxes)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'var': f'{loss_dict.get("var", 0):.4f}',
                'topo': f'{loss_dict.get("topo", 0):.4f}'
            })
    
    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, optimizer, scheduler,
          num_epochs, device, save_path, sam_model=None):
    """
    Full training loop with validation and checkpointing.
    """
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, sam_model
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_metrics['val_loss'])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, save_path / 'best.pth')
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
              f'Val mIoU={val_metrics["mIoU"]:.4f}')
    
    return model