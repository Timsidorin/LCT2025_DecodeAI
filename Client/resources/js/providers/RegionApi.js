import {BaseApi} from "./BaseApi.js";

export class RegionApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async getListRegion(includeCities) {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/regions';
            super.params = {
                include_cities: includeCities,
            }
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
