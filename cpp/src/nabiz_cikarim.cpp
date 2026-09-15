// NABIZ — uç C++ çıkarım ikili dosyası (ONNX Runtime C++ API).
//
// Kapsam (bilinçli sınırlandırma — bkz. ../../README.md "Faz 4"):
//   Girdi ÖZNİTELİK VEKTÖRÜ (CSV satırı, virgülle ayrılmış float) alır — ham
//   WAV'dan mel çıkarımı burada YOK (Python dsp/features.py'de). Bu ikili,
//   "ONNX Runtime C++ API ile uç çıkarım" iddiasını kanıtlar: model yükleme,
//   tensor kurulumu, çıkarım, gecikme ölçümü, Python'la golden-vector paritesi.
//
// Kullanım:
//   nabiz_cikarim.exe <model.onnx> <girdi.csv> [cikti.csv]
//     girdi.csv : her satır bir pencere, N virgüllü float (başlık YOK)
//     cikti.csv : verilirse anomali skorları oraya yazılır; yoksa stdout
//
// Anomali skoru = yeniden yapılandırma MSE'si (autoencoder çıktısı vs girdi)
// — nabiz.edge.onnx_disa.yeniden_yapilandirma_skoru ile aynı tanım.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::vector<float>> csvOku(const std::string& yol) {
    std::ifstream dosya(yol);
    if (!dosya) throw std::runtime_error("CSV açılamadı: " + yol);
    std::vector<std::vector<float>> satirlar;
    std::string satir;
    while (std::getline(dosya, satir)) {
        if (satir.empty()) continue;
        std::vector<float> deger;
        std::stringstream ss(satir);
        std::string hucre;
        while (std::getline(ss, hucre, ',')) {
            deger.push_back(std::stof(hucre));
        }
        satirlar.push_back(std::move(deger));
    }
    return satirlar;
}

double yuzdelik(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(v.size() - 1));
    return v[idx];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Kullanim: " << argv[0] << " <model.onnx> <girdi.csv> [cikti.csv]\n";
        return 1;
    }
    const std::string modelYolu = argv[1];
    const std::string girdiYolu = argv[2];
    const std::string ciktiYolu = (argc >= 4) ? argv[3] : "";

    std::vector<std::vector<float>> satirlar;
    try {
        satirlar = csvOku(girdiYolu);
    } catch (const std::exception& e) {
        std::cerr << "HATA: " << e.what() << "\n";
        return 1;
    }
    if (satirlar.empty()) {
        std::cerr << "HATA: girdi CSV boş\n";
        return 1;
    }
    const size_t n = satirlar.size();
    const size_t boyut = satirlar[0].size();
    for (const auto& s : satirlar) {
        if (s.size() != boyut) {
            std::cerr << "HATA: tüm satırlar aynı öznitelik sayısına sahip olmalı\n";
            return 1;
        }
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nabiz_cikarim");
        Ort::SessionOptions secenekler;
        secenekler.SetIntraOpNumThreads(1);
        secenekler.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring genisModelYolu(modelYolu.begin(), modelYolu.end());
        Ort::Session oturum(env, genisModelYolu.c_str(), secenekler);
#else
        Ort::Session oturum(env, modelYolu.c_str(), secenekler);
#endif

        Ort::AllocatorWithDefaultOptions ayirici;
        Ort::AllocatedStringPtr girdiAdPtr = oturum.GetInputNameAllocated(0, ayirici);
        Ort::AllocatedStringPtr ciktiAdPtr = oturum.GetOutputNameAllocated(0, ayirici);
        const char* girdiAdlari[] = {girdiAdPtr.get()};
        const char* ciktiAdlari[] = {ciktiAdPtr.get()};

        Ort::MemoryInfo bellekBilgisi =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<double> skorlar(n);
        std::vector<double> gecikmelerMs;
        gecikmelerMs.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            std::vector<float> girdi = satirlar[i];  // Run() boyunca canlı kalmalı
            std::array<int64_t, 2> sekil{1, static_cast<int64_t>(boyut)};

            const auto baslangic = std::chrono::high_resolution_clock::now();

            Ort::Value girdiTensor = Ort::Value::CreateTensor<float>(
                bellekBilgisi, girdi.data(), girdi.size(), sekil.data(), sekil.size());

            auto ciktilar = oturum.Run(Ort::RunOptions{nullptr}, girdiAdlari, &girdiTensor, 1,
                                        ciktiAdlari, 1);

            const auto bitis = std::chrono::high_resolution_clock::now();
            gecikmelerMs.push_back(
                std::chrono::duration<double, std::milli>(bitis - baslangic).count());

            const float* yeniden = ciktilar[0].GetTensorData<float>();
            double mse = 0.0;
            for (size_t j = 0; j < boyut; ++j) {
                const double fark = static_cast<double>(yeniden[j]) - static_cast<double>(girdi[j]);
                mse += fark * fark;
            }
            skorlar[i] = mse / static_cast<double>(boyut);
        }

        if (!ciktiYolu.empty()) {
            std::ofstream cikti(ciktiYolu);
            cikti.precision(9);
            for (double s : skorlar) cikti << s << "\n";
        } else {
            std::cout.precision(9);
            for (double s : skorlar) std::cout << s << "\n";
        }

        std::cerr << "n=" << n << "  boyut=" << boyut
                   << "  P50=" << yuzdelik(gecikmelerMs, 50) << " ms"
                   << "  P95=" << yuzdelik(gecikmelerMs, 95) << " ms\n";
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime HATASI: " << e.what() << "\n";
        return 1;
    }
}
