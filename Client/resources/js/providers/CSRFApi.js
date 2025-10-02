import {BaseApi} from "./BaseApi.js";

export class CSRFApi extends BaseApi{
    constructor() {
        super(__BASE__LARAVEL__URL__);
    }

    async getToken() {
        try {
            super.httpMethod = 'get';
            super.sourceUrl = '/api/token';
            let response = await super.createRequest();
            localStorage.setItem('csrf', response.data)
        } catch (e) {
            return e;
        }
    }
}
