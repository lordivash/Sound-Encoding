#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

using Byte = unsigned char;

class Encoder {
public:
  virtual int encode(const std::vector<float>& samples, int channels, float eps, int from_pos, Byte* buf, int udp_payload_capacity) = 0;
  virtual ~Encoder() = default;
};

class PrefixBitmaskEncoder : public Encoder {
public:
  // Channels must be from 1 to 255
  // udp_payload_capacity should be enough to write custom header
  int encode(const std::vector<float>& samples, int channels, float eps, int from_pos, Byte* buf, int udp_payload_capacity) override {
    auto [cur_pos, zero_colons] = skip_prefix_zeros(samples, channels, eps, from_pos);

    std::memcpy(buf, &zero_colons, sizeof(std::int16_t));

    int used_bytes{ sizeof(std::int16_t) };
    int remaining_bytes{ udp_payload_capacity - used_bytes };

    cur_pos = encode_bitmask_payload(samples, channels, eps, cur_pos, buf + used_bytes, remaining_bytes);

    return cur_pos;
  }

private:
  std::pair<int, std::int16_t> skip_prefix_zeros(const std::vector<float>& samples, int channels, float eps, int from_pos) {
    int cur_pos{ from_pos };
    std::int16_t zero_colons{ 0 };

    while (cur_pos < samples.size()) {
      for (int i{ cur_pos }; i < cur_pos + channels; ++i) {
        if (std::abs(samples[i]) > eps) {
          return { cur_pos, zero_colons };
        }
      }
      cur_pos += channels;
      ++zero_colons;
    }

    return { cur_pos, zero_colons };
  }

  std::tuple<int, int> get_payload_dims(const std::vector<float>& samples, int channels, float eps, int from_pos, Byte* bitmask, int bitmask_bytes, int bytes_availible) {
    int float_capacity{ static_cast<int>(bytes_availible / sizeof(float)) };

    int active_channels{ 0 };
    int cur_float_cnt{ 0 };

    int cur_pos{ from_pos };
    int cols_passed{ 0 };

    while (cur_pos < samples.size()) {
      Byte added_bits[32] = { 0 };
      int to_add{ 0 };

      for (int i{ cur_pos }, channel{ 0 }, byte{ 0 }, bit{ 0 }; i < cur_pos + channels; ++i, ++channel) {
        if (std::abs(samples[i]) > eps && !(bitmask[byte] & (1 << bit))) {
          added_bits[byte] |= (1 << bit);
          ++active_channels;
          to_add += cols_passed;
        }

        ++bit;

        if (bit >= 8) {
          ++byte;
          bit = 0;
        }
      }

      if (cur_float_cnt + to_add + active_channels > float_capacity) {
        return { cur_pos, cols_passed };
      }

      ++cols_passed;
      cur_float_cnt += (to_add + active_channels);
      cur_pos += channels;
      for (int i{ 0 }; i < bitmask_bytes; ++i) bitmask[i] |= added_bits[i];
    }

    return { cur_pos, cols_passed };
  }

  int encode_bitmask_payload(const std::vector<float>& samples, int channels, float eps, int from_pos, Byte* buf, int bytes_availible) {
    int bitmask_bytes{ (channels + 7) / 8 };
    Byte bitmask[32] = { 0 };

    int payload_bytes{ bytes_availible - bitmask_bytes };
    auto [end_pos, cols] { get_payload_dims(samples, channels, eps, from_pos, bitmask, bitmask_bytes, payload_bytes) };

    std::memcpy(buf, bitmask, bitmask_bytes);
    buf += bitmask_bytes;

    for (int i{ 0 }, cur_pos{ from_pos }; i < cols; ++i, cur_pos += channels) {
      for (int j{ cur_pos }, channel{ 0 }, byte{ 0 }, bit{ 0 }; j < cur_pos + channels; ++j, ++channel) {
        if (bitmask[byte] & (1 << bit)) {
          std::memcpy(buf, &samples[j], sizeof(float));
          buf += sizeof(float);
        }

        ++bit;
        if (bit >= 8) {
          ++byte;
          bit = 0;
        }
      }
    }

    return end_pos;
  }
};
#endif