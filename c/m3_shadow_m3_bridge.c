#include "m3_shadow_m3_bridge.h"

#include <string.h>

int m3_shd_m3_make_request(const m3_shd_m3_load_snapshot_t *snapshot,
                           m3_shd_request_t *request,
                           m3_shd_candidate_t *candidate) {
    if (!snapshot || !request || !candidate || !snapshot->model_id
        || !snapshot->demand
        || !snapshot->bytes || !snapshot->drive_resource_id
        || snapshot->key.layer < 0 || snapshot->key.expert < 0) return -1;
    memset(request, 0, sizeof(*request));
    request->request_id = snapshot->request_id;
    request->model_id = snapshot->model_id;
    request->key = snapshot->key;
    request->bytes = snapshot->bytes;
    request->consumer_need_ns = snapshot->consumer_need_ns;
    request->demand = snapshot->demand;
    request->forward_id = snapshot->forward_id;
    request->generation = snapshot->generation;
    request->consumer_node = snapshot->consumer_node;

    memset(candidate, 0, sizeof(*candidate));
    candidate->candidate_id = snapshot->drive_resource_id;
    candidate->n_resources = 1;
    candidate->resources[0] = snapshot->drive_resource_id;
    candidate->available = 1;
    return 0;
}
