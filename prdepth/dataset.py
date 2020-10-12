import numpy as np
import tensorflow as tf


class Dataset:

    def __init__(
            self, lfile, bsz, height, width,
            niter=0, isval=False, aug=True, seed=0):
        '''
        Args:
            lfile: Path of file with list of image file names.
            bsz: Batch size you want this to generate.
            niter: Resume at niterations.
            isval: Running on train or val.
                (random crops and shuffling for train).
            aug: Data augmentation or not.
        '''

        self.bsz = bsz
        self.height = height
        self.width = width
        self.isrand = not isval
        self.aug = aug

        # Setup fetch graph
        self.graph()

        # Load file list
        if isinstance(lfile, str):
            self.files = [line.strip() for line in open(lfile).readlines()]
        else:
            self.files = lfile
        self.ndata = len(self.files)
        self.niter = niter * bsz

        # Setup shuffling
        if self.isrand:
            self.rand = np.random.RandomState(seed)
            idx = self.rand.permutation(self.ndata)
            for i in range(niter // self.ndata):
                idx = self.rand.permutation(self.ndata)
            self.idx = np.int32(idx)
        else:
            self.idx = np.int32(np.arange(self.ndata))

    def graph(self):
        self.names = []
        # Create placeholders
        for i in range(self.bsz):
            self.names.append(tf.placeholder(tf.string))

        dbatch, ibatch = [], []
        for i in range(self.bsz):
            depth = tf.read_file(self.names[i] + '_f.png')
            depth = tf.image.decode_png(depth, channels=1, dtype=tf.uint16)
            image = tf.read_file(self.names[i] + '_i.png')
            image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)

            depth = tf.to_float(depth) / (2**16 - 1.0)
            image = tf.to_float(image) / 255.0

            imsz = tf.shape(depth)
            ro_bound = np.pi * 5.0 / 180.0

            if not self.isrand:
                yoff = (imsz[0] - self.height) // 2
                xoff = (imsz[1] - self.width) // 2
                depth = depth[yoff:yoff + self.height,
                              xoff:xoff + self.width, :]
                image = image[yoff:yoff + self.height,
                              xoff:xoff + self.width, :]

            elif not self.aug:
                yoff = tf.random_uniform(
                    [], 0, imsz[0] - self.height + 1, dtype=tf.int32)
                xoff = tf.random_uniform(
                    [], 0, imsz[1] - self.width + 1, dtype=tf.int32)
                depth = depth[yoff:yoff + self.height,
                              xoff:xoff + self.width, :]
                image = image[yoff:yoff + self.height,
                              xoff:xoff + self.width, :]

            elif self.isrand and self.aug:
                # random scaling
                di = tf.concat([depth, image], axis=-1)
                scale = tf.random_uniform([], 1.0, 1.5, dtype=tf.float32)
                hheight = tf.to_int32(scale * tf.to_float(imsz[0]))
                wwidth = tf.to_int32(scale * tf.to_float(imsz[1]))
                di = tf.image.resize_bilinear(
                    di[tf.newaxis], [hheight, wwidth])

                # random rotation
                ro = tf.random_uniform(
                    [], -ro_bound, ro_bound, dtype=tf.float32)
                di = tf.contrib.image.rotate(di[0], ro, 'NEAREST')

                # random crop
                di = tf.random_crop(di, [self.height, self.width, 4])

                # random flip
                di = tf.image.random_flip_left_right(di)

                depth, image = di[:, :, :1], di[:, :, 1:]

                # random brightness, contrast, saturation, hue
                image = tf.image.random_brightness(image, 0.4)
                image = tf.image.random_contrast(image, 0.6, 1.4)
                image = tf.image.random_saturation(image, 0.6, 1.4)

                # correct depth for scaling
                depth = depth / scale

            dbatch.append(depth)
            ibatch.append(image)

        dbatch = tf.stack(dbatch)
        ibatch = tf.stack(ibatch) * 255.0

        # Fetch op
        self.dbatch = tf.Variable(
            tf.zeros([self.bsz, self.height, self.width, 1], dtype=tf.float32),
            trainable=False)
        self.ibatch = tf.Variable(
            tf.zeros([self.bsz, self.height, self.width, 3], dtype=tf.float32),
            trainable=False)
        self.fetch_op = [tf.assign(self.dbatch, dbatch).op,
                        tf.assign(self.ibatch, ibatch).op]

    def fdict(self):
        fd = {}
        for i in range(self.bsz):
            idx = self.idx[self.niter % self.ndata]
            self.niter = self.niter + 1
            if self.niter % self.ndata == 0 and self.isrand:
                self.idx = np.int32(self.rand.permutation(self.ndata))
            fd[self.names[i]] = self.files[idx]
        return fd

    # Sets up a common batch variable for train and val and ops
    # to swap in pre-fetched image data.
    def tvSwap(self, vset):
        dbatch = tf.Variable(
            tf.zeros(self.dbatch.shape, dtype=tf.float32), trainable=False)
        ibatch = tf.Variable(
            tf.zeros(self.ibatch.shape, dtype=tf.float32), trainable=False)
        tSwap = [tf.assign(dbatch, tf.identity(self.dbatch)).op,
                 tf.assign(ibatch, tf.identity(self.ibatch)).op]
        vSwap = [tf.assign(dbatch, tf.identity(vset.dbatch)).op,
                 tf.assign(ibatch, tf.identity(vset.ibatch)).op]

        return dbatch, ibatch, tSwap, vSwap
