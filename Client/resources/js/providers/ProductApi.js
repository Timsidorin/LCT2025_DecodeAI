import {BaseApi} from "./BaseApi.js";

export class ProductApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async getListProduct() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/stats/products';
            return super.createRequest();
        } catch (e) {
            return e;
        }
    }
}
