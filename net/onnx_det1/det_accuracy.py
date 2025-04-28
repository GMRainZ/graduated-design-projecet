    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # targets (tensor): (n_gt_all_batch, [img_index clsid cx cy l s theta gaussian_θ_labels]) θ ∈ [-pi/2, pi/2)
        # shapes (tensor): (b, [(h_raw, w_raw), (hw_ratios, wh_paddings)])
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, theta

        # NMS
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        out = non_max_suppression_obb(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls) # list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out): # pred (tensor): (n, [xylsθ, conf, cls])
            labels = targets[targets[:, 0] == si, 1:7] # labels (tensor):(n_gt, [clsid cx cy l s theta]) θ[-pi/2, pi/2)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0] # shape (tensor): (h_raw, w_raw)
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                # pred[:, 5] = 0
                pred[:, 6] = 0
            poly = rbox2poly(pred[:, :5]) # (n, 8)
            pred_poly = torch.cat((poly, pred[:, -2:]), dim=1) # (n, [poly, conf, cls])
            hbbox = xywh2xyxy(poly2hbb(pred_poly[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbb = torch.cat((hbbox, pred_poly[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) 

            pred_polyn = pred_poly.clone() # predn (tensor): (n, [poly, conf, cls])
            scale_polys(im[si].shape[1:], pred_polyn[:, :8], shape, shapes[si][1])  # native-space pred
            hbboxn = xywh2xyxy(poly2hbb(pred_polyn[:, :8])) # (n, [x1 y1 x2 y2])
            pred_hbbn = torch.cat((hbboxn, pred_polyn[:, -2:]), dim=1) # (n, [xyxy, conf, cls]) native-space pred
            

            # Evaluate
            if nl:
                # tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                tpoly = rbox2poly(labels[:, 1:6]) # target poly
                tbox = xywh2xyxy(poly2hbb(tpoly)) # target  hbb boxes [xyxy]
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labels_hbbn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels (n, [cls xyxy])
                correct = process_batch(pred_hbbn, labels_hbbn, iouv)
                if plots:
                    confusion_matrix.process_batch(pred_hbbn, labels_hbbn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred_poly[:, 8].cpu(), pred_poly[:, 9].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt: # just save hbb pred results!
                save_one_txt(pred_hbbn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
                # LOGGER.info('The horizontal prediction results has been saved in txt, which format is [cls cx cy w h /conf/]')
            if save_json: # save hbb pred results and poly pred results.
                save_one_json(pred_hbbn, pred_polyn, jdict, path, class_map)  # append to COCO-JSON dictionary
                # LOGGER.info('The hbb and obb results has been saved in json file')
            callbacks.run('on_val_image_end', pred_hbb, pred_hbbn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(im, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(im, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # LOGGER.info("----------------------------------------------")
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # LOGGER.info("----------------------------------------------")


    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)