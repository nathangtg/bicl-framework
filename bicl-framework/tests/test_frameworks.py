import torch
from src.frameworks import EnhancedEWC, EnhancedUnifiedFramework
from src.model import get_model

def test_ewc_loss_increases_with_divergence():
    model = get_model()
    ewc = EnhancedEWC(model, lambda_reg=100.0)
    
    # Mock Fisher and optimal params
    ewc.fisher_info = {'0.weight': torch.ones_like(model[0].weight)}
    ewc.optimal_params = {'0.weight': model[0].weight.clone().detach()}
    ewc.task_count = 1

    base_loss = torch.tensor(1.0)
    ewc_loss1 = ewc.ewc_loss(base_loss)
    
    # Change the model's weights
    with torch.no_grad():
        model[0].weight += 1.0
    
    ewc_loss2 = ewc.ewc_loss(base_loss)

    # Assert that the EWC penalty made the loss higher
    assert ewc_loss2 > ewc_loss1

def test_unified_framework_memory_buffer_capacity():
    model = get_model()
    framework = EnhancedUnifiedFramework(model, memory_buffer_size=10)
    
    # Add more samples than the buffer can hold
    x_data = torch.randn(20, 100)
    y_data = torch.randint(0, 10, (20,))
    framework.add_to_memory(x_data, y_data)
    
    # Assert that the buffer did not exceed its capacity
    assert len(framework.memory_buffer) == 10