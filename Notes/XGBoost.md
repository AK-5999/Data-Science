<img width="1403" height="656" alt="image" src="https://github.com/user-attachments/assets/14491551-24de-4ff0-b71d-815e51585ea0" />


<img width="1374" height="668" alt="image" src="https://github.com/user-attachments/assets/a50bb429-e6e9-473e-852c-5e4be5d347b2" />

<img width="1388" height="516" alt="image" src="https://github.com/user-attachments/assets/c7a5abb2-9b3a-4509-a6e6-ad8ea9941edd" />


| Component                   | Tumhari Understanding         | Correct Version                                                                                                        |
| --------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Omega: L2 ((\lambda))**   | Weight shrink karta hai       | ✅ Bilkul sahi                                                                                                          |
| **Omega: L1 ((\alpha))**    | Kuch weights ko 0 karta hai   | ✅ Bilkul sahi                                                                                                          |
| **Omega: Gamma ((\gamma))** | Cost of split                 | ✅ Bilkul sahi                                                                                                          |
| **Gain: L2**                | Aggressive split ko rokta hai | ⚠️ Indirectly. Actually gain calculation me leaf scores ko shrink karta hai, jisse weak splits ka gain kam ho jata hai |
| **Gain: Gamma**             | Cost of split                 | ✅ Bilkul sahi                                                                                                          |
| **Leaf Value**              | Sirf L2 use hota hai          | ⚠️ Common derivation me haan, but full XGBoost me L1 bhi use ho sakta hai                                              |
