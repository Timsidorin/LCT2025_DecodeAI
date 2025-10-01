import {BaseApi} from "./BaseApi.js";
import {getFirstCharInUp} from "../utils/mix.js";

export class ProductApi extends BaseApi{
    constructor() {
        super(__BASE__PYTHON__URL__);
    }

    async getListProduct() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/dashboard/stats/products';
            let data = await super.createRequest();
            data.data.products_analysis.map((element) => {
                element.product = getFirstCharInUp(element.product);
            });
            return data;
        } catch (e) {
            return e;
        }
    }
}
