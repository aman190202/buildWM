  Basic shapes                                                                                                                                                                                    
  1. torch.zeros(3, 4)
  2. torch.randn(2, 3, 4)                                                                                                                                                                         
  3. torch.arange(10)                                       
  4. torch.linspace(0, 1, 5)

  Indexing
  5. a = torch.randn(10, 2); a[3]
  6. a = torch.randn(10, 2); a[2:5]
  7. a = torch.randn(10, 2); a[:, 0]
  8. a = torch.randn(1000, 2); idx = torch.randint(0, 1000, (64,)); a[idx]

  Reshaping
  9. a = torch.randn(64); a[:, None]
  10. a = torch.randn(64); a[None, :]
  11. a = torch.randn(64, 1) * torch.randn(1, 32)
  12. a = torch.randn(64, 2); b = torch.randn(64, 32); torch.cat([a, b], dim=-1)

  Math on tensors
  13. a = torch.tensor([4.0, 9.0, 16.0]); torch.sqrt(a)
  14. a = torch.randn(64); torch.cumprod(a, dim=0)
  15. a = torch.randn(64, 2); b = torch.randn(64, 1); a * b
  16. a = torch.randn(64, 2); b = torch.randn(64, 2); (a - b) ** 2

  Neural network shapes
  17. nn.Linear(34, 128)(torch.randn(64, 34))
  18. nn.Linear(128, 2)(torch.randn(64, 128))
  19. F.relu(torch.randn(64, 128))

  Putting it together
  20. x = torch.randn(64, 2); t = torch.randint(0, 1000, (64,)); torch.cat([x, t[:, None].float()], dim=-1)


  1. [3,4]
  2. [2,3,4]
  3. [10,]
  4. [5,]
  5. [2,]
  6. [3,2]
  7. [10]
  8. [64,2]
  9. [64,1]
  10. [1,64]
  11. [64,32]
  12. [64,34]
  13. [3]
  14. [64]
  15. [64,2]
  16. [64,2]
  17. [64,128]
  18. [64,2]
  19. [64,128]
  20. [64,3]