# IA Modules Documentation Index

Welcome to the **ia_modules** documentation! This index provides quick access to all documentation resources.

---

## 🚀 Getting Started

Start here if you're new to ia_modules:

1. **[Getting Started Guide](./GETTING_STARTED.md)** - Your first steps with ia_modules
2. **[IA Modules Guide](./IA_MODULES_GUIDE.md)** - Complete overview of the framework
3. **[Features Overview](./FEATURES.md)** - What ia_modules can do

---

## 📚 Core Documentation

### Database & SQL

- **[SQL Translation System](./SQL_TRANSLATION.md)** ⭐ **NEW!** - Complete guide to database-agnostic SQL
- **[SQL Quick Reference](./SQL_QUICK_REFERENCE.md)** ⭐ **NEW!** - Quick reference for SQL translations
- **[Database Interfaces](./DATABASE_INTERFACES.md)** - Database abstraction layer
- **[Database System Research](./DATABASE_SYSTEM_RESEARCH.md)** - Design decisions and research

### Pipeline System

- **[Pipeline Architecture](./PIPELINE_ARCHITECTURE.md)** - Core pipeline concepts and design
- **[Execution Architecture](./EXECUTION_ARCHITECTURE.md)** - How pipelines execute
- **[Cyclic Graphs](./CYCLIC_GRAPHS.md)** - Building pipelines with cycles and loops
- **[Test Pipelines Guide](./TEST_PIPELINES_GUIDE.md)** - Example pipelines and patterns

### Agent System

- **[Agent System Explained](./AGENT_SYSTEM_EXPLAINED.md)** - Multi-agent orchestration
- **[Comparison: LangChain vs LangGraph](./COMPARISON_LANGCHAIN_LANGGRAPH.md)** - How ia_modules compares

### Human-in-the-Loop

- **[Human-in-the-Loop Comprehensive Guide](./HUMAN_IN_LOOP_COMPREHENSIVE.md)** - Complete HITL documentation

---

## 🛠️ Development

### For Developers

- **[Developer Guide](./DEVELOPER_GUIDE.md)** - Contributing and extending ia_modules
- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Testing Guide](./TESTING_GUIDE.md)** - How to test your code

### Advanced Features

- **[Plugin System](./PLUGIN_SYSTEM_DOCUMENTATION.md)** - Creating and using plugins
- **[CLI Tool](./CLI_TOOL_DOCUMENTATION.md)** - Command-line interface
- **[Checkpointing](./CHECKPOINTING_DESIGN.md)** - State persistence and recovery
- **[Streaming Support](./STREAMING_SUPPORT_PLAN.md)** - Real-time data streaming

---

## 🔒 Reliability & Observability

### Reliability Framework

- **[Reliability Usage Guide](./RELIABILITY_USAGE_GUIDE.md)** - How to use reliability features
- **[Reliability Framework Reference](./RELIABILITY_FRAMEWORK_REFERENCE.md)** - Technical reference
- **[Enterprise Reliability Framework](./ENTERPRISE_RELIABILITY_FRAMEWORK_REFERENCE.md)** - Enterprise features

### RAG (Retrieval-Augmented Generation)

- **[RAG Design Plan](./RAG_DESIGN_PLAN.md)** - RAG system architecture

---

## 🎯 Key Features

### SQL Translation (NEW!)

ia_modules provides **automatic SQL translation** from PostgreSQL syntax to all major databases:

```python
# Write once in PostgreSQL syntax
db.execute("""
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255),
        is_active BOOLEAN DEFAULT TRUE,
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    )
""")
# Works on PostgreSQL, MySQL, MSSQL, and SQLite!
```

**Learn more:**
- [SQL Translation System](./SQL_TRANSLATION.md) - Complete guide
- [SQL Quick Reference](./SQL_QUICK_REFERENCE.md) - Cheat sheet

### Multi-Agent Pipelines

Build complex workflows with multiple AI agents:
- [Agent System Explained](./AGENT_SYSTEM_EXPLAINED.md)
- [Pipeline Architecture](./PIPELINE_ARCHITECTURE.md)

### Database Abstraction

Connect to any database with a unified interface:
- PostgreSQL
- MySQL
- MSSQL
- SQLite

**Learn more:** [Database Interfaces](./DATABASE_INTERFACES.md)

---

## 📖 Philosophy

- **[Doctrine of Autonomous Platforms](./Doctrine_of_Autonomous_Platforms.md)** - Design philosophy

---

## 🔍 Quick Links

### By Topic

| Topic | Documents |
|-------|-----------|
| **SQL & Databases** | [SQL Translation](./SQL_TRANSLATION.md), [SQL Quick Ref](./SQL_QUICK_REFERENCE.md), [DB Interfaces](./DATABASE_INTERFACES.md) |
| **Pipelines** | [Architecture](./PIPELINE_ARCHITECTURE.md), [Execution](./EXECUTION_ARCHITECTURE.md), [Cyclic Graphs](./CYCLIC_GRAPHS.md) |
| **Agents** | [Agent System](./AGENT_SYSTEM_EXPLAINED.md), [Comparison](./COMPARISON_LANGCHAIN_LANGGRAPH.md) |
| **Development** | [Dev Guide](./DEVELOPER_GUIDE.md), [API Ref](./API_REFERENCE.md), [Testing](./TESTING_GUIDE.md) |
| **Reliability** | [Usage Guide](./RELIABILITY_USAGE_GUIDE.md), [Framework Ref](./RELIABILITY_FRAMEWORK_REFERENCE.md) |
| **Advanced** | [Plugins](./PLUGIN_SYSTEM_DOCUMENTATION.md), [Checkpointing](./CHECKPOINTING_DESIGN.md), [Streaming](./STREAMING_SUPPORT_PLAN.md) |

### By Experience Level

**Beginner:**
1. [Getting Started](./GETTING_STARTED.md)
2. [IA Modules Guide](./IA_MODULES_GUIDE.md)
3. [SQL Quick Reference](./SQL_QUICK_REFERENCE.md)
4. [Test Pipelines Guide](./TEST_PIPELINES_GUIDE.md)

**Intermediate:**
1. [Pipeline Architecture](./PIPELINE_ARCHITECTURE.md)
2. [SQL Translation System](./SQL_TRANSLATION.md)
3. [Agent System Explained](./AGENT_SYSTEM_EXPLAINED.md)
4. [Reliability Usage Guide](./RELIABILITY_USAGE_GUIDE.md)

**Advanced:**
1. [Developer Guide](./DEVELOPER_GUIDE.md)
2. [API Reference](./API_REFERENCE.md)
3. [Checkpointing Design](./CHECKPOINTING_DESIGN.md)
4. [Database System Research](./DATABASE_SYSTEM_RESEARCH.md)

---

## 🆕 What's New

### Latest Additions

- ⭐ **[SQL Translation System](./SQL_TRANSLATION.md)** - Complete database-agnostic SQL
- ⭐ **[SQL Quick Reference](./SQL_QUICK_REFERENCE.md)** - Quick reference guide
- 📝 **[Execution Architecture](./EXECUTION_ARCHITECTURE.md)** - Updated execution model

---

## 🤝 Contributing

Want to improve the documentation? See:
- [Developer Guide](./DEVELOPER_GUIDE.md) - How to contribute
- [Testing Guide](./TESTING_GUIDE.md) - How to test your changes

---

## 📬 Need Help?

1. Check the relevant documentation above
2. Look at example pipelines in [Test Pipelines Guide](./TEST_PIPELINES_GUIDE.md)
3. Review the [API Reference](./API_REFERENCE.md)
4. See [Getting Started](./GETTING_STARTED.md) for basics

---

**Happy coding with ia_modules!** 🚀
