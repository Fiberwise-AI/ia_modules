import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Search, MessageSquare, Clock, User, Bot, TrendingUp } from 'lucide-react';

export default function ConversationHistory({ sessionId }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchType, setSearchType] = useState('semantic');
  const queryClient = useQueryClient();

  // Fetch conversation history
  const { data: messages, isLoading: messagesLoading } = useQuery({
    queryKey: ['conversation', sessionId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:5555/api/memory/${sessionId}?limit=50`);
      if (!response.ok) throw new Error('Failed to fetch conversation history');
      return response.json();
    },
    enabled: !!sessionId
  });

  // Fetch memory stats
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['memory-stats', sessionId],
    queryFn: async () => {
      const response = await fetch(`http://localhost:5555/api/memory/${sessionId}/stats`);
      if (!response.ok) throw new Error('Failed to fetch memory stats');
      return response.json();
    },
    enabled: !!sessionId
  });

  // Search memory
  const searchMutation = useMutation({
    mutationFn: async ({ query, type }) => {
      const response = await fetch('http://localhost:5555/api/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          search_type: type,
          session_id: sessionId,
          limit: 20
        })
      });
      if (!response.ok) throw new Error('Search failed');
      return response.json();
    }
  });

  const handleSearch = () => {
    if (searchQuery.trim()) {
      searchMutation.mutate({ query: searchQuery, type: searchType });
    }
  };

  // Ensure displayMessages is always an array
  const displayMessages = Array.isArray(searchMutation.data?.results) 
    ? searchMutation.data.results 
    : Array.isArray(messages) 
    ? messages 
    : [];

  if (!sessionId) {
    return (
      <div className="p-8 text-center text-gray-500">
        <MessageSquare className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p>Select an execution to view conversation history</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with Stats */}
      <div className="border-b bg-gray-50 p-4">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <MessageSquare className="w-5 h-5" />
          Conversation History
        </h2>

        {/* Stats Cards */}
        {!statsLoading && stats && (
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-white p-3 rounded-lg border">
              <div className="text-sm text-gray-600">Total Messages</div>
              <div className="text-2xl font-bold text-blue-600">{stats.total_messages}</div>
            </div>
            <div className="bg-white p-3 rounded-lg border">
              <div className="text-sm text-gray-600">Total Tokens</div>
              <div className="text-2xl font-bold text-purple-600">
                {stats.total_tokens?.toLocaleString() || 0}
              </div>
            </div>
            <div className="bg-white p-3 rounded-lg border">
              <div className="text-sm text-gray-600">Duration</div>
              <div className="text-2xl font-bold text-green-600">
                {stats.first_message && stats.last_message
                  ? `${Math.round((new Date(stats.last_message) - new Date(stats.first_message)) / 60000)}m`
                  : 'N/A'}
              </div>
            </div>
          </div>
        )}

        {/* Search Bar */}
        <div className="flex gap-2">
          <div className="flex-1 flex gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search conversation..."
              className="flex-1 px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <select
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="semantic">Semantic</option>
              <option value="keyword">Keyword</option>
            </select>
          </div>
          <button
            onClick={handleSearch}
            disabled={searchMutation.isPending}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
          >
            <Search className="w-4 h-4" />
            Search
          </button>
        </div>

        {searchMutation.data && (
          <div className="mt-2 text-sm text-gray-600">
            Found {searchMutation.data.results?.length || 0} results
            <button
              onClick={() => searchMutation.reset()}
              className="ml-2 text-blue-600 hover:text-blue-700"
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Messages List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messagesLoading ? (
          <div className="text-center py-8 text-gray-500">Loading messages...</div>
        ) : displayMessages.length === 0 ? (
          <div className="text-center py-8 text-gray-500">No messages found</div>
        ) : (
          displayMessages.map((message, idx) => (
            <MessageCard key={idx} message={message} />
          ))
        )}
      </div>
    </div>
  );
}

function MessageCard({ message }) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div
      className={`p-4 rounded-lg border ${
        isUser
          ? 'bg-blue-50 border-blue-200'
          : isSystem
          ? 'bg-gray-50 border-gray-200'
          : 'bg-purple-50 border-purple-200'
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isUser ? (
            <User className="w-4 h-4 text-blue-600" />
          ) : isSystem ? (
            <TrendingUp className="w-4 h-4 text-gray-600" />
          ) : (
            <Bot className="w-4 h-4 text-purple-600" />
          )}
          <span className="font-medium text-sm capitalize">{message.role}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <Clock className="w-3 h-3" />
          {new Date(message.timestamp).toLocaleString()}
        </div>
      </div>

      {/* Content */}
      <div className="text-sm text-gray-700 whitespace-pre-wrap">{message.content}</div>

      {/* Metadata */}
      {message.metadata && Object.keys(message.metadata).length > 0 && (
        <details className="mt-2">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
            Metadata
          </summary>
          <pre className="mt-1 text-xs bg-white p-2 rounded border overflow-x-auto">
            {JSON.stringify(message.metadata, null, 2)}
          </pre>
        </details>
      )}

      {/* Token Count */}
      {message.token_count && (
        <div className="mt-2 text-xs text-gray-500">
          Tokens: {message.token_count.toLocaleString()}
        </div>
      )}
    </div>
  );
}
