function train_model_manually!(model, dataloader; learning_rate=DEFAULTS[:LEARNING_RATE])
    loss_fn(y_hat, y) = Flux.crossentropy(y_hat, y)
    optimizer = Flux.Adam(learning_rate)

    total_loss = 0.0
    num_batches = 0

    for (x_batch, y_batch) in dataloader
        # Compute loss inside gradient calculation
        loss, grads = Flux.withgradient(model) do m
            y_hat = m(x_batch)
            loss_fn(y_hat, y_batch)
        end
        
        # Apply gradients
        Flux.Optimise.update!(optimizer, Flux.params(model), grads)

        # Track loss
        total_loss += loss
        num_batches += 1
    end

    return total_loss / num_batches  # Return average loss
end

for epoch in 1:100
    loss = train_model!(mo, trainloader_mnist, learning_rate=0.05)
    accuracy = calc_accuracy(mo, testloader_mnist)
    println("Epoch = $(epoch) : Loss = $(loss) : Accuracy = $(accuracy)")
end
