import argparse
import os
import time
import torch

from valid import test
from utils import util


class Train():
    def __init__(self, args):
        # Make saving directory---------------------------------------------------------------------
        save_dir_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(save_dir_path, exist_ok=True)
        self.save_dir_path = save_dir_path

        # Logger instance----------------------------------------------------------
        self.logger = util.Logger(save_dir_path)
        self.logger.info('-' * 10)
        self.logger.info(vars(args))

    def train(self, model, criterion, optimizer, scheduler, dataloader, num_epochs, device, args):
        # Schedule learning rate--------------------------------------------
        def adjust_lr(epoch):
            step_size = 40
            lr = args.lr * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        # ++++++++++++++++++++++++++++++++++start++++++++++++++++++++++++++++++++++
        start_time = time.time()
        for epoch in range(num_epochs):
            self.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

            model.train()
            adjust_lr(epoch)

            # Training----------------------------------------------------------
            running_loss = 0.0
            batch_num = 0
            for data in dataloader:
                inputs, labels, _ = data
                batch_num += 1

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # with torch.set_grad_enabled(True):
                outputs = model(inputs)

                # Sum up the stripe softmax loss
                loss = 0
                if isinstance(outputs, (list,)):
                    for logits in outputs:
                        stripe_loss = criterion(logits, labels)
                        loss += stripe_loss
                elif isinstance(outputs, (torch.Tensor,)):
                    loss = criterion(outputs, labels)
                else:
                    raise Exception('outputs type is error !')

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            # ------------------------------------------------------------------

            epoch_loss = running_loss / len(dataloader.dataset)
            self.logger.info('Training Loss: {:.4f}'.format(epoch_loss))

            # Save result to self.logger----------------------------------------------------------
            self.logger.x_epoch_loss.append(epoch + 1)
            self.logger.y_train_loss.append(epoch_loss)

            # test ----------------------------------------------------------
            if (epoch + 1) % args.test_every == 0 or epoch + 1 == num_epochs:
                # Testing / Validating
                torch.cuda.empty_cache()
                CMC, mAP = test(model, args.dataset, args.dataset_path, 512)
                self.logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' % (CMC[0], CMC[4], CMC[9], mAP))

                self.logger.x_epoch_test.append(epoch + 1)
                self.logger.y_test['top1'].append(CMC[0])
                self.logger.y_test['mAP'].append(mAP)
                if epoch + 1 != num_epochs:
                    util.save_network(model, self.save_dir_path, str(epoch + 1))
            self.logger.info('-' * 10)
            # ---------------------------------------------------------------------------

        # ++++++++++++++++++++++++++++++++++start epoch end ++++++++++++++++++++++++++++++++++

        # Save the loss curve---------------------------------------------------------------------------
        self.logger.save_curve()

        # save running time ---------------------------------------------------------------------------
        time_elapsed = time.time() - start_time
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save final model weights---------------------------------------------------------------------
        util.save_network(model, self.save_dir_path, 'final')
