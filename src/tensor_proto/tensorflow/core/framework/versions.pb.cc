// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/versions.proto

#include "tensorflow/core/framework/versions.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace tensorflow {
class VersionDefDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<VersionDef> _instance;
} _VersionDef_default_instance_;
}  // namespace tensorflow
static void InitDefaultsscc_info_VersionDef_tensorflow_2fcore_2fframework_2fversions_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_VersionDef_default_instance_;
    new (ptr) ::tensorflow::VersionDef();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::VersionDef::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_VersionDef_tensorflow_2fcore_2fframework_2fversions_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_VersionDef_tensorflow_2fcore_2fframework_2fversions_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_tensorflow_2fcore_2fframework_2fversions_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fframework_2fversions_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fframework_2fversions_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_tensorflow_2fcore_2fframework_2fversions_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::VersionDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::VersionDef, producer_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::VersionDef, min_consumer_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::VersionDef, bad_consumers_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::VersionDef)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_VersionDef_default_instance_),
};

const char descriptor_table_protodef_tensorflow_2fcore_2fframework_2fversions_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n(tensorflow/core/framework/versions.pro"
  "to\022\ntensorflow\"K\n\nVersionDef\022\020\n\010producer"
  "\030\001 \001(\005\022\024\n\014min_consumer\030\002 \001(\005\022\025\n\rbad_cons"
  "umers\030\003 \003(\005Bn\n\030org.tensorflow.frameworkB"
  "\016VersionsProtosP\001Z=github.com/tensorflow"
  "/tensorflow/tensorflow/go/core/framework"
  "\370\001\001b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_sccs[1] = {
  &scc_info_VersionDef_tensorflow_2fcore_2fframework_2fversions_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_once;
static bool descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto = {
  &descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_initialized, descriptor_table_protodef_tensorflow_2fcore_2fframework_2fversions_2eproto, "tensorflow/core/framework/versions.proto", 251,
  &descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_once, descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_sccs, descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fframework_2fversions_2eproto::offsets,
  file_level_metadata_tensorflow_2fcore_2fframework_2fversions_2eproto, 1, file_level_enum_descriptors_tensorflow_2fcore_2fframework_2fversions_2eproto, file_level_service_descriptors_tensorflow_2fcore_2fframework_2fversions_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_tensorflow_2fcore_2fframework_2fversions_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_tensorflow_2fcore_2fframework_2fversions_2eproto), true);
namespace tensorflow {

// ===================================================================

void VersionDef::InitAsDefaultInstance() {
}
class VersionDef::_Internal {
 public:
};

VersionDef::VersionDef()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.VersionDef)
}
VersionDef::VersionDef(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena),
  bad_consumers_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.VersionDef)
}
VersionDef::VersionDef(const VersionDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      bad_consumers_(from.bad_consumers_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::memcpy(&producer_, &from.producer_,
    static_cast<size_t>(reinterpret_cast<char*>(&min_consumer_) -
    reinterpret_cast<char*>(&producer_)) + sizeof(min_consumer_));
  // @@protoc_insertion_point(copy_constructor:tensorflow.VersionDef)
}

void VersionDef::SharedCtor() {
  ::memset(&producer_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&min_consumer_) -
      reinterpret_cast<char*>(&producer_)) + sizeof(min_consumer_));
}

VersionDef::~VersionDef() {
  // @@protoc_insertion_point(destructor:tensorflow.VersionDef)
  SharedDtor();
}

void VersionDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
}

void VersionDef::ArenaDtor(void* object) {
  VersionDef* _this = reinterpret_cast< VersionDef* >(object);
  (void)_this;
}
void VersionDef::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void VersionDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const VersionDef& VersionDef::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_VersionDef_tensorflow_2fcore_2fframework_2fversions_2eproto.base);
  return *internal_default_instance();
}


void VersionDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.VersionDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  bad_consumers_.Clear();
  ::memset(&producer_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&min_consumer_) -
      reinterpret_cast<char*>(&producer_)) + sizeof(min_consumer_));
  _internal_metadata_.Clear();
}

const char* VersionDef::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // int32 producer = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          producer_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int32 min_consumer = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          min_consumer_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated int32 bad_consumers = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedInt32Parser(_internal_mutable_bad_consumers(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24) {
          _internal_add_bad_consumers(::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* VersionDef::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.VersionDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 producer = 1;
  if (this->producer() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->_internal_producer(), target);
  }

  // int32 min_consumer = 2;
  if (this->min_consumer() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(2, this->_internal_min_consumer(), target);
  }

  // repeated int32 bad_consumers = 3;
  {
    int byte_size = _bad_consumers_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteInt32Packed(
          3, _internal_bad_consumers(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.VersionDef)
  return target;
}

size_t VersionDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.VersionDef)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated int32 bad_consumers = 3;
  {
    size_t data_size = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      Int32Size(this->bad_consumers_);
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _bad_consumers_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // int32 producer = 1;
  if (this->producer() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_producer());
  }

  // int32 min_consumer = 2;
  if (this->min_consumer() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_min_consumer());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void VersionDef::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.VersionDef)
  GOOGLE_DCHECK_NE(&from, this);
  const VersionDef* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<VersionDef>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.VersionDef)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.VersionDef)
    MergeFrom(*source);
  }
}

void VersionDef::MergeFrom(const VersionDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.VersionDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  bad_consumers_.MergeFrom(from.bad_consumers_);
  if (from.producer() != 0) {
    _internal_set_producer(from._internal_producer());
  }
  if (from.min_consumer() != 0) {
    _internal_set_min_consumer(from._internal_min_consumer());
  }
}

void VersionDef::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.VersionDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void VersionDef::CopyFrom(const VersionDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.VersionDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool VersionDef::IsInitialized() const {
  return true;
}

void VersionDef::InternalSwap(VersionDef* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  bad_consumers_.InternalSwap(&other->bad_consumers_);
  swap(producer_, other->producer_);
  swap(min_consumer_, other->min_consumer_);
}

::PROTOBUF_NAMESPACE_ID::Metadata VersionDef::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::VersionDef* Arena::CreateMaybeMessage< ::tensorflow::VersionDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::VersionDef >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
