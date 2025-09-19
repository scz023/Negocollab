    def generate_object_center(self,
                               cav_contents,
                               reference_lidar_pose,
                               enlarge_z=False):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        enlarge_z :
            if True, enlarge the z axis range to include more object

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        tmp_object_dict = {}
        for cav_content in cav_contents:
            tmp_object_dict.update(cav_content['params']['vehicles'])

        output_dict = {}
        filter_range = self.params['anchor_args']['cav_lidar_range'] \
            if self.train else self.params['gt_range']

        box_utils.project_world_objects(tmp_object_dict,
                                        output_dict,
                                        reference_lidar_pose,
                                        filter_range,
                                        self.params['order'],
                                        enlarge_z)

        object_np = np.zeros((self.params['max_num'], 7))
        mask = np.zeros(self.params['max_num'])
        object_ids = []

        for i, (object_id, object_bbx) in enumerate(output_dict.items()):
            # modify here to avoid index out of bounds in onject_np
            if i >= self.params['max_num']:
                break
            object_np[i] = object_bbx[0, :]
            mask[i] = 1
            object_ids.append(object_id)
        return object_np, mask, object_ids