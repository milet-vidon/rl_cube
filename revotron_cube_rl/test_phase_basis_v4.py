import unittest
import torch
from phase_basis_policy_v4 import PHASES, SCHEMA, basis, PhaseBasisPolicy, restored


class BasisPolicyTests(unittest.TestCase):
    def model(self):
        heads = []
        for j in range(len(PHASES)):
            coefficients = torch.zeros((7, 28), dtype=torch.float64)
            coefficients[0] = j*.01
            coefficients[1] = .1
            coefficients[6] = .2
            heads.append((torch.tensor([.5]), coefficients))
        return PhaseBasisPolicy(torch.full((28,), -2.), torch.full((28,), 2.), heads)

    def test_exact_basis_and_phase_selection(self):
        model = self.model()
        x = torch.zeros((2, 198)); x[0, 185] = 1; x[1, 186] = 1; x[:, -1] = .75
        self.assertTrue(torch.allclose(model(x)[0], torch.full((28,), .125, dtype=torch.float64)))
        self.assertTrue(torch.allclose(model(x)[1], torch.full((28,), .135, dtype=torch.float64)))

    def test_state_and_previous_target_features_cannot_affect_prediction(self):
        model = self.model()
        x = torch.randn((4, 198)); x[:, 185:197] = 0; x[:, 189] = 1; x[:, -1] = .3
        changed = x.clone(); changed[:, :185] = torch.randn((4, 185))*10000
        self.assertTrue(torch.equal(model(x), model(changed)))

    def test_increment_and_limits(self):
        model = self.model()
        x = torch.zeros((1, 198)); x[:, 185] = 1; x[:, -1] = 1
        result = model.targets(x, torch.zeros((1, 28)))
        self.assertTrue(torch.equal(result, torch.full((1, 28), .04, dtype=torch.float64)))

    def test_restore_exact(self):
        model = self.model()
        saved = {'metadata': {'schema': SCHEMA}, 'model': model.state_dict()}
        copy = restored(saved)
        x = torch.zeros((1, 198)); x[:, 185] = 1; x[:, -1] = .42
        self.assertTrue(torch.equal(model(x), copy(x)))


if __name__ == '__main__':
    unittest.main()
