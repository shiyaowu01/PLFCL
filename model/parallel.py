"""Parallel PLFCL calls with portable, unwrapped checkpoint keys."""

from torch import nn


class PLFCLDataParallel(nn.DataParallel):
    """Route custom model methods through DataParallel's scatter and gather."""

    def get_loss(self, pc, feat, img, loader, center, M, cube, cam_para, gt):
        return self(pc, feat, img, loader, center, M, cube, cam_para,
                    operation='loss', gt=gt).mean()

    def sample(self, pc, feat, img, loader, center, M, cube, cam_para, h=5, step=20):
        return self(pc, feat, img, loader, center, M, cube, cam_para,
                    operation='sample', h=h, step=step)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)
